from .fx_utils import find_getitem_maybe
from .multi_output_match import MultiOutputMatch
from .vllm_inductor_pass import VllmInductorPass
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload
from typing import Callable, NamedTuple, Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
import torch
import torch._inductor.pattern_matcher as pm
logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()

def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device='cuda')

def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device='cuda')
RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

class QuantKey(NamedTuple):
    """
    Named tuple for identifying the type of quantization.
    dtype: quantized data type
    static: static quantization if True, dynamic if False
    group_shape: quantization group shape
    symmetric: symmetric if True, asymmetric if False

    TODO(luka) use QuantDescriptor once standardized:
    https://github.com/vllm-project/vllm/issues/8913

    """
    dtype: torch.dtype
    static: bool
    group_shape: GroupShape
    symmetric: bool = True

    def __str__(self):
        group_shape = 'per_tensor' if self.group_shape == GroupShape.PER_TENSOR else 'per_token' if self.group_shape == GroupShape.PER_TOKEN else str(self.group_shape)
        return f"QuantKey({('static' if self.static else 'dynamic')},{fx.graph.dtype_abbrs[self.dtype]},{group_shape},{('a' if not self.symmetric else '')}symmetric)"
kFp8StaticTensorSym = QuantKey(FP8_DTYPE, True, GroupShape.PER_TENSOR, True)
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, False, GroupShape.PER_TENSOR, True)
kFp8DynamicTokenSym = QuantKey(FP8_DTYPE, False, GroupShape.PER_TOKEN, True)
QUANT_OPS: dict[QuantKey, OpOverload] = {kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default, kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default, kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default}

class FusedRMSQuantKey(NamedTuple):
    """
    Named tuple for identifying the type of RMSNorm + quant fusion.
    quant: type of quantization
    fused_add: does the op also perform the residual add
    """
    quant: QuantKey
    fused_add: bool

    def __str__(self):
        return f"FusedQuantKey({self.quant}, with{('' if self.fused_add else 'out')} residual)"
FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {FusedRMSQuantKey(kFp8StaticTensorSym, False): torch.ops._C.rms_norm_static_fp8_quant.default, FusedRMSQuantKey(kFp8StaticTensorSym, True): torch.ops._C.fused_add_rms_norm_static_fp8_quant.default, FusedRMSQuantKey(kFp8DynamicTokenSym, False): torch.ops._C.rms_norm_dynamic_per_token_quant.default, FusedRMSQuantKey(kFp8DynamicTokenSym, True): torch.ops._C.rms_norm_dynamic_per_token_quant.default}

class QuantMultiOutputMatch(MultiOutputMatch):

    def __init__(self, match: pm.Match, quant_op, fused_op):
        super().__init__(match)
        assert isinstance(quant_op, OpOverload)
        assert isinstance(fused_op, OpOverload)
        self.QUANT_OP = quant_op
        self.FUSED_OP = fused_op

    def insert_fused_node(self, fused_return_mapping: dict[int, tuple[fx.Node, int]], **kwargs):
        """
        This utility function inserts an auto-functionalized node for FUSED_OP.
        It also correctly sets its meta value and rebinds the users of the
        unfused nodes to use the fused node instead.

        :param fused_return_mapping: A dictionary, mapping from getitem indices
        of the fused node result to a tuple of the old node and a getitem index.
        :param kwargs: kwargs that get directly forwarded to the auto_fn node

        Example:
        If we want to replace this graph:
        _, x1, x2 = auto_fn(op1)
        _, y1, y2 = auto_fn(op2)

        with
        _, x1, y2, x2 = auto_fn(FUSED_OP)

        we would call:
        insert_fused_node({1: (op1_node, 1), 2: (op2_node, 2), 3: (op1_node, 2)}

        Note that the 0th element is None for auto-functionalized in-place ops.
        Hence, others appear 1-indexed.
        """
        fused_node = self.insert_auto_fn(self.FUSED_OP, kwargs)
        indices = fused_return_mapping.keys()
        getitem_nodes = self.insert_getitems(fused_node, indices)
        meta_val = [None] * (max(indices) + 1)
        for idx, getitem_node in zip(indices, getitem_nodes):
            old_node, old_idx = fused_return_mapping[idx]
            old_getitem = find_getitem_maybe(old_node, old_idx)
            if old_getitem is not None:
                old_getitem.replace_all_uses_with(getitem_node)
                getitem_node.meta['val'] = old_getitem.meta['val']
            meta_val[idx] = old_node.meta['val'][old_idx]
        fused_node.meta['val'] = tuple(meta_val)

class RMSNormQuantPattern:

    def __init__(self, epsilon: float, key: FusedRMSQuantKey):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        assert key.quant in QUANT_OPS, f'unsupported quantization scheme {key.quant}'
        self.QUANT_OP = QUANT_OPS[key.quant]
        assert key in FUSED_OPS, f'unsupported fused rmsnorm+quant op for {key}'
        self.FUSED_OP = FUSED_OPS[key]

class RMSNormStaticQuantPattern(RMSNormQuantPattern):

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        fused_key = FusedRMSQuantKey(fused_add=False, quant=QuantKey(dtype=quant_dtype, static=True, group_shape=GroupShape.PER_TENSOR, symmetric=symmetric))
        super().__init__(epsilon, fused_key)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at1 = auto_functionalized(RMS_OP, result=result_rms, input=input, weight=weight, epsilon=self.epsilon)
            at2 = auto_functionalized(self.QUANT_OP, result=result, input=at1[1], scale=scale)
            return at2[1]

        def replacement(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP, result=result, input=input, weight=weight, scale=scale, epsilon=self.epsilon)
            return at[1]
        inputs = [torch.empty(5, 4, device='cuda', dtype=self.quant_dtype), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(1, 5), empty_fp32(1, 1)]
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)

class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        key = FusedRMSQuantKey(fused_add=True, quant=QuantKey(dtype=quant_dtype, static=True, group_shape=GroupShape.PER_TENSOR, symmetric=symmetric))
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass, record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(RMS_ADD_OP, input=input, residual=residual, weight=weight, epsilon=self.epsilon)
            at1 = auto_functionalized(self.QUANT_OP, result=result, input=at[1], scale=scale)
            return (at1[1], at[2])

        def replacement(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP, result=result, input=input, residual=residual, weight=weight, scale=scale, epsilon=self.epsilon)
            return (at[1], at[2])
        inputs = [torch.empty(5, 4, device='cuda', dtype=self.quant_dtype), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(1, 5), empty_fp32(1, 1)]
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass, extra_check=lambda m: record_match(self.Match(m, self.QUANT_OP, self.FUSED_OP)))

    class Match(QuantMultiOutputMatch):

        def process(self):
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)
            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 1
            with self.inserting_after_match():
                kwargs = self.match.kwargs.copy()
                fused_return_mapping = {1: (quant_node, 1), 2: (rms_node, 2)}
                self.insert_fused_node(fused_return_mapping, **kwargs, epsilon=rms_node.kwargs['epsilon'])

class RMSNormDynamicQuantPattern(RMSNormQuantPattern):

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, group_shape: GroupShape=GroupShape.PER_TOKEN, symmetric=True):
        key = FusedRMSQuantKey(fused_add=False, quant=QuantKey(dtype=quant_dtype, static=False, group_shape=group_shape, symmetric=symmetric))
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass, record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at1 = auto_functionalized(RMS_OP, result=result_rms, input=input, weight=weight, epsilon=self.epsilon)
            at2 = auto_functionalized(self.QUANT_OP, result=result, input=at1[1], scale=scale, scale_ub=None)
            return (at2[1], at2[2])

        def replacement(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP, result=result, input=input, weight=weight, scale=scale, epsilon=self.epsilon, scale_ub=None, residual=None)
            return (at[1], at[2])
        inputs = [torch.empty(5, 4, device='cuda', dtype=self.quant_dtype), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(1, 5), empty_fp32(1, 1)]
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass, extra_check=lambda m: record_match(self.Match(m, self.QUANT_OP, self.FUSED_OP)))

    class Match(QuantMultiOutputMatch):

        def process(self):
            rms_node = self.find_auto_fn(RMS_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)
            assert len(rms_node.users) == 1
            assert len(quant_node.users) == 2
            with self.inserting_after_match():
                kwargs = self.match.kwargs.copy()
                del kwargs['result_rms']
                fused_return_mapping = {1: (quant_node, 1), 2: (quant_node, 2)}
                self.insert_fused_node(fused_return_mapping, epsilon=rms_node.kwargs['epsilon'], scale_ub=None, residual=None, **kwargs)

class FusedAddRMSNormDynamicQuantPattern(RMSNormQuantPattern):

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, group_shape: GroupShape=GroupShape.PER_TOKEN, symmetric=True):
        key = FusedRMSQuantKey(fused_add=True, quant=QuantKey(dtype=quant_dtype, static=False, group_shape=group_shape, symmetric=symmetric))
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass, record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(RMS_ADD_OP, input=input, residual=residual, weight=weight, epsilon=self.epsilon)
            at1 = auto_functionalized(self.QUANT_OP, result=result, input=at[1], scale=scale, scale_ub=None)
            return (at1[1], at[2], at1[2])

        def replacement(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP, result=result, input=input, weight=weight, scale=scale, epsilon=self.epsilon, scale_ub=None, residual=residual)
            return (at[1], at[3], at[2])
        inputs = [torch.empty(5, 4, device='cuda', dtype=self.quant_dtype), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(1, 5), empty_fp32(1, 1)]
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass, extra_check=lambda m: record_match(self.Match(m, self.QUANT_OP, self.FUSED_OP)))

    class Match(QuantMultiOutputMatch):

        def process(self):
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(self.QUANT_OP)
            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 2
            with self.inserting_after_match():
                kwargs = self.match.kwargs.copy()
                fused_return_mapping = {1: (quant_node, 1), 2: (quant_node, 2), 3: (rms_node, 2)}
                self.insert_fused_node(fused_return_mapping, epsilon=rms_node.kwargs['epsilon'], scale_ub=None, **kwargs)

class FusionPass(VllmInductorPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.
    It also manually processes multi-output matches, as those are broken in
    the torch pattern matcher.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """
    _instance: 'Optional[FusionPass]' = None

    @classmethod
    def instance(cls, config: VllmConfig):
        """
        Get the singleton instance of the FusionPass.
        If the instance exists, the config is updated but
        initialization is not repeated.
        """
        if cls._instance is None:
            cls._instance = FusionPass(config)
        else:
            cls._instance.pass_config = config.compilation_config.pass_config
        return cls._instance

    def __init__(self, config: VllmConfig):
        assert self.__class__._instance is None, 'FusionPass singleton instance already exists'
        super().__init__(config)
        self.matches: list[MultiOutputMatch] = []
        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name='fusion_pass')
        for epsilon in [1e-05, 1e-06]:
            RMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)
            FusedAddRMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(self.patterns, self.record_match)
            RMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns, self.record_match)
            FusedAddRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns, self.record_match)
            torch._inductor.pattern_matcher._seen_patterns.clear()

    def record_match(self, match: MultiOutputMatch) -> bool:
        self.matches.append(match)
        return False

    def process_matches(self, graph: fx.Graph):
        """
        Manually process multi-output matches and replace them with fused nodes.
        See MultiOutputMatch for more details.
        """
        for match in self.matches:
            match.process()
        graph.eliminate_dead_code()
        assert all((node not in graph.nodes for match in self.matches for node in match.match.nodes))

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, 'before_fusion')
        count = self.patterns.apply(graph)
        logger.debug('Replaced %s patterns', count)
        self.dump_graph(graph, 'after_pattern_match')
        self.process_matches(graph)
        logger.debug('Post-processed %s matches', len(self.matches))
        self.dump_graph(graph, 'after_fusion')
        self.matches.clear()
        self.end_and_log()