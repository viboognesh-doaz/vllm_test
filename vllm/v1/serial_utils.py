from collections.abc import Sequence
from inspect import isclass
from msgspec import msgpack
from types import FunctionType
from typing import Any, Optional, Union
from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.inputs import BaseMultiModalField, MultiModalBatchedField, MultiModalFieldConfig, MultiModalFieldElem, MultiModalFlatField, MultiModalKwargs, MultiModalKwargsItem, MultiModalSharedField, NestedTensors
import cloudpickle
import dataclasses
import numpy as np
import pickle
import torch
import zmq
logger = init_logger(__name__)
CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {MultiModalFlatField: 'flat', MultiModalSharedField: 'shared', MultiModalBatchedField: 'batched'}
bytestr = Union[bytes, bytearray, memoryview, zmq.Frame]

def _log_insecure_serialization_warning():
    logger.warning_once('Allowing insecure serialization using pickle due to VLLM_ALLOW_INSECURE_SERIALIZATION=1')

class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent 
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(self, size_threshold: Optional[int]=None):
        if size_threshold is None:
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        self.aux_buffers: Optional[list[bytestr]] = None
        self.size_threshold = size_threshold
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b'']
            bufs[0] = self.encoder.encode(obj)
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ('O', 'V'):
            return self._encode_ndarray(obj)
        if isinstance(obj, slice):
            return tuple((int(v) if v is not None else None for v in (obj.start, obj.stop, obj.step)))
        if isinstance(obj, MultiModalKwargs):
            mm: MultiModalKwargs = obj
            if not mm.modalities:
                return dict(mm)
            return [[{'modality': elem.modality, 'key': elem.key, 'data': self._encode_nested_tensors(elem.data), 'field': self._encode_mm_field(elem.field)} for elem in item.values()] for itemlist in mm._items_by_modality.values() for item in itemlist]
        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(f'Object of type {type(obj)} is not serializableSet VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow fallback to pickle-based serialization.')
        if isinstance(obj, FunctionType):
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))
        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_ndarray(self, obj: np.ndarray) -> tuple[str, tuple[int, ...], Union[int, memoryview]]:
        assert self.aux_buffers is not None
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)
        return (obj.dtype.str, obj.shape, data)

    def _encode_tensor(self, obj: torch.Tensor) -> tuple[str, tuple[int, ...], Union[int, memoryview]]:
        assert self.aux_buffers is not None
        arr = obj.flatten().contiguous().view(torch.uint8).numpy()
        if obj.nbytes < self.size_threshold:
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr.data)
        else:
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype).removeprefix('torch.')
        return (dtype, obj.shape, data)

    def _encode_nested_tensors(self, nt: NestedTensors) -> Any:
        if isinstance(nt, torch.Tensor):
            return self._encode_tensor(nt)
        if isinstance(nt, (int, float)):
            return nt
        return [self._encode_nested_tensors(x) for x in nt]

    def _encode_mm_field(self, field: BaseMultiModalField):
        name = MMF_CLASS_TO_FACTORY.get(field.__class__)
        if not name:
            raise TypeError(f'Unsupported field type: {field.__class__}')
        field_values = (getattr(field, f.name) for f in dataclasses.fields(field))
        return (name, *field_values)

class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Optional[Any]=None):
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(*args, ext_hook=self.ext_hook, dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    def decode(self, bufs: Union[bytestr, Sequence[bytestr]]) -> Any:
        if isinstance(bufs, (bytes, bytearray, memoryview, zmq.Frame)):
            return self.decoder.decode(bufs)
        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is slice:
                return slice(*obj)
            if issubclass(t, MultiModalKwargs):
                if isinstance(obj, list):
                    return MultiModalKwargs.from_items(self._decode_mm_items(obj))
                return MultiModalKwargs({k: self._decode_nested_tensors(v) for k, v in obj.items()})
        return obj

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        buffer = self.aux_buffers[data] if isinstance(data, int) else bytearray(data)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer:
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        return arr.view(torch_dtype).view(shape)

    def _decode_mm_items(self, obj: list) -> list[MultiModalKwargsItem]:
        decoded_items = []
        for item in obj:
            elems = []
            for v in item:
                v['data'] = self._decode_nested_tensors(v['data'])
                factory_meth_name, *field_args = v['field']
                factory_meth = getattr(MultiModalFieldConfig, factory_meth_name)
                if factory_meth_name == 'flat':
                    field_args[0] = self._decode_nested_slices(field_args[0])
                v['field'] = factory_meth(None, *field_args).field
                elems.append(MultiModalFieldElem(**v))
            decoded_items.append(MultiModalKwargsItem.from_elems(elems))
        return decoded_items

    def _decode_nested_tensors(self, obj: Any) -> NestedTensors:
        if isinstance(obj, (int, float)):
            return obj
        if not isinstance(obj, list):
            raise TypeError(f'Unexpected NestedTensors contents: {type(obj)}')
        if obj and isinstance(obj[0], str):
            return self._decode_tensor(obj)
        return [self._decode_nested_tensors(x) for x in obj]

    def _decode_nested_slices(self, obj: Any) -> Any:
        assert isinstance(obj, (list, tuple))
        if obj and (not isinstance(obj[0], (list, tuple))):
            return slice(*obj)
        return [self._decode_nested_slices(x) for x in obj]

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            if code == CUSTOM_TYPE_PICKLE:
                return pickle.loads(data)
            if code == CUSTOM_TYPE_CLOUDPICKLE:
                return cloudpickle.loads(data)
        raise NotImplementedError(f'Extension type code {code} is not supported')