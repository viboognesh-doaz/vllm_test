from bitblas import Matmul, auto_detect_nvidia_target
from bitblas import MatmulConfig
from bitblas.cache import get_database_path, global_operator_cache
from bitblas.quantization.utils import general_compress
import bitblas
from typing import Optional
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.bitblas_utils import BITBLAS_OPTIMIZE_FEATURES, BITBLAS_SUPPORTED_GROUP_SIZES, MINIMUM_BITBLAS_VERSION, bitblas_make_empty_g_idx, bitblas_sort_g_idx, check_bitblas_supports_shape, query_bitblas_supported_quant_types, unpack_gptq_qweight, unpack_gptq_qzeros
from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
logger = init_logger(__name__)

class BitBLASLinearKernel(MPLinearKernel):
    OPT_FEATURES: list[int] = BITBLAS_OPTIMIZE_FEATURES
    ENABLE_TUNING: bool = True
    MATMUL_LAYOUT: str = 'nt'
    BITBLAS_DTYPES: dict[torch.dtype, str] = {torch.float32: 'float32', torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.half: 'float16', torch.int8: 'int8'}
    bitblas_matmul: object = None

    def __init__(self, c: MPLinearLayerConfig, w_q_param_name: str, w_s_param_name: str, w_zp_param_name: Optional[str]=None, w_gidx_param_name: Optional[str]=None, bitblas_quant_config: Optional[QuantizationConfig]=None):
        self.quant_config = bitblas_quant_config
        super().__init__(c, w_q_param_name, w_s_param_name, w_zp_param_name, w_gidx_param_name)

    def repack_bitblas_from_gptq(self, b_q_weight: torch.Tensor, scales: torch.Tensor, qzeros: Optional[torch.Tensor]=None):
        assert self.bitblas_matmul is not None, 'bitblas_matmul is None'
        quant_config = self.quant_config
        qweight = b_q_weight.T.contiguous().view(quant_config.torch_storage_dtype)
        intweight = unpack_gptq_qweight(qweight, quant_config.weight_bits).contiguous()
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(intweight.cpu()).cuda()
        scales = scales.T.contiguous()
        if qzeros is None:
            return (qweight, scales, None)
        weight_bits = quant_config.weight_bits
        intzeros = unpack_gptq_qzeros(qzeros, weight_bits).T.contiguous()
        zeros: Optional[torch.Tensor] = None
        zeros_mode = self.bitblas_matmul.config.zeros_mode
        if zeros_mode == 'original':
            zeros = intzeros.to(torch.float16).contiguous()
        elif zeros_mode == 'rescale':
            assert zeros is not None, 'zeros should not be None'
            zeros[:, :] = intzeros.to(torch.float16)[:, :] * scales[:, :]
        elif zeros_mode == 'quantized':
            zeros = torch.Tensor(general_compress(intzeros.T.contiguous().cpu().numpy(), weight_bits)).to(qweight.device).to(quant_config.torch_storage_dtype).contiguous()
        else:
            raise ValueError('Unsupported zeros type: {}'.format(zeros_mode))
        return (qweight, scales, zeros)

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        is_bitblas_installed = True
        try:
            if bitblas.__version__ < MINIMUM_BITBLAS_VERSION:
                raise ImportError(f'bitblas version is wrong. Please install bitblas>={MINIMUM_BITBLAS_VERSION}')
        except ImportError:
            is_bitblas_installed = False
        if not is_bitblas_installed:
            return (False, f'bitblas is not installed. Please install bitblas by running `pip install bitblas>={MINIMUM_BITBLAS_VERSION}`')
        quant_types = query_bitblas_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (False, f'Quant type ({c.weight_type}) not supported by  BitBLAS, supported types are: {quant_types}')
        if c.group_size not in BITBLAS_SUPPORTED_GROUP_SIZES:
            return (False, f'Group size ({c.group_size}) not supported by BitBLAS, supported group sizes are: {BITBLAS_SUPPORTED_GROUP_SIZES}')
        return check_bitblas_supports_shape(c.partition_weight_shape[1], c.partition_weight_shape[0], c.full_weight_shape[0], c.group_size)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        quant_config = self.quant_config
        if self.w_gidx_name is None:
            self.w_gidx_name = 'g_idx'
        if self.w_zp_name is None:
            self.w_zp_name = 'qzeros'
        if c.has_g_idx:
            g_idx, g_idx_sort_indices = bitblas_sort_g_idx(getattr(layer, self.w_gidx_name))
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, bitblas_make_empty_g_idx(device))
            layer.g_idx_sort_indices = bitblas_make_empty_g_idx(device)
        if c.zero_points:
            raise NotImplementedError('Zero points not supported by BitBLAS')
        else:
            setattr(layer, self.w_zp_name, bitblas_make_empty_g_idx(device))
        bitblas_qweight, bitblas_scales, bitblas_qzeros = self.repack_bitblas_from_gptq(layer.qweight, layer.scales, None if quant_config.is_sym else layer.qzeros)
        replace_parameter(layer, self.w_q_name, bitblas_qweight)
        replace_parameter(layer, self.w_s_name, bitblas_scales)
        if bitblas_qzeros is not None:
            replace_parameter(layer, self.w_zp_name, bitblas_qzeros)

    def configure_bitblas_matmul(self, infeatures: int, outfeatures: int, params_dtype: torch.dtype, bias: bool) -> None:
        enable_tuning = self.ENABLE_TUNING
        layout = self.MATMUL_LAYOUT
        bits = self.quant_config.weight_bits
        self._configure_bitblas_matmul(infeatures, outfeatures, params_dtype, enable_tuning, bias, layout, bits)

    def _configure_bitblas_matmul(self, infeatures, outfeatures, params_dtype, enable_tuning, bias, layout, bits):
        bitblas_dtype = self.BITBLAS_DTYPES[params_dtype]
        quant_config = self.quant_config
        with_scaling = False
        with_zeros = False
        group_size = quant_config.group_size
        zeros_mode = quant_config.zeros_mode
        if quant_config.quant_method == 'gptq':
            with_scaling = True
            with_zeros = True
            W_dtype = f'uint{bits}'
            if quant_config.is_sym:
                with_zeros = False
                W_dtype = f'int{bits}'
        else:
            raise ValueError(f'Unsupported quant_method {quant_config.quant_method}')
        matmul_config = MatmulConfig(M=self.OPT_FEATURES, N=outfeatures, K=infeatures, A_dtype=bitblas_dtype, W_dtype=W_dtype, out_dtype=bitblas_dtype, accum_dtype='int32' if bitblas_dtype == 'int8' else bitblas_dtype, storage_dtype=quant_config.storage_dtype, with_scaling=with_scaling, with_zeros=with_zeros, group_size=group_size, with_bias=bias, layout=layout, zeros_mode=zeros_mode)
        self.bitblas_matmul = self._get_or_create_bitblas_operator(matmul_config, enable_tuning)

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        BITBLAS_DATABASE_PATH = get_database_path()
        BITBLAS_TARGET = auto_detect_nvidia_target()
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                TUNING_MESSAGE = f'BitBLAS Operator {config} tuned and saved to database.'
                logger.info(TUNING_MESSAGE)
            else:
                _message = f'BitBLAS Operator {config} created without tuning. '
                logger.info(_message)
        else:
            _message = f'BitBLAS Operator {config} retrieved from cache.'
            logger.info(_message)
        return bitblas_matmul

    def apply_gptq_bitblas_linear(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        output_size_per_partition = self.config.partition_weight_shape[1]
        out_shape = x.shape[:-1] + (output_size_per_partition,)
        args = [x, layer.qweight, layer.scales]
        if self.bitblas_matmul.config.with_zeros:
            args.append(layer.qzeros)
        output = self.bitblas_matmul(*args)
        return output.view(out_shape)

    def apply_weights(self, layer, x, bias=None):
        NOT_IMPLEMENT_MESSAGE = f'{self.__class__.__name__}.apply_weights is not implemented. Please use BitBLASLinearKernel.apply_gptq_bitblas_linear instead'
        raise NotImplementedError(NOT_IMPLEMENT_MESSAGE)