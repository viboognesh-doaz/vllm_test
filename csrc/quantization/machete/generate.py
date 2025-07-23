from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import reduce
from typing import Optional, Union
from vllm_cutlass_library_extension import DataType, EpilogueScheduleTag, EpilogueScheduleType, MixedInputKernelScheduleType, TileSchedulerTag, TileSchedulerType, VLLMDataType, VLLMDataTypeNames, VLLMDataTypeSize, VLLMDataTypeTag, VLLMDataTypeTorchDataTypeTag, VLLMDataTypeVLLMScalarTypeTag, VLLMKernelScheduleTag
import itertools
import jinja2
import math
import os
import shutil
DISPATCH_TEMPLATE = '\n#include "../machete_mm_launcher.cuh"\n\nnamespace machete {\n\n{% for impl_config in impl_configs %}\n{% set type_sig = gen_type_sig(impl_config.types) -%}\n{% for s in impl_config.schedules %}\nextern torch::Tensor impl_{{type_sig}}_sch_{{gen_sch_sig(s)}}(MMArgs);\n{%- endfor %}\n\ntorch::Tensor mm_dispatch_{{type_sig}}(MMArgs args) {\n  [[maybe_unused]] auto M = args.A.size(0);\n  [[maybe_unused]] auto N = args.B.size(1);\n  [[maybe_unused]] auto K = args.A.size(1);\n    \n  if (!args.maybe_schedule) {\n    {%- for cond, s in impl_config.heuristic %}\n    {%if cond is not none%}if ({{cond}})\n    {%- else %}else\n    {%- endif %}\n        return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);{% endfor %}\n  }\n\n  {%- for s in impl_config.schedules %}\n  if (*args.maybe_schedule == "{{ gen_sch_sig(s) }}")\n    return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);\n  {%- endfor %}\n  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "\n                                     "schedule = ", *args.maybe_schedule);\n}\n{%- endfor %}\n\n\nstatic inline std::optional<at::ScalarType> maybe_scalartype(\n    std::optional<at::Tensor> const& t) {\n    if (!t) {\n      return std::nullopt;\n    } else {\n      return t->scalar_type();\n    };\n}\n\ntorch::Tensor mm_dispatch(MMArgs args) {\n  auto out_type = args.maybe_out_type.value_or(args.A.scalar_type());\n  auto a_type = args.A.scalar_type();\n  auto maybe_g_scales_type = maybe_scalartype(args.maybe_group_scales);\n  auto maybe_g_zeros_type = maybe_scalartype(args.maybe_group_zeros);\n  auto maybe_ch_scales_type = maybe_scalartype(args.maybe_channel_scales);\n  auto maybe_tok_scales_type = maybe_scalartype(args.maybe_token_scales);\n\n  {% for impl_config in impl_configs %}\n  {% set t = impl_config.types -%}\n  {% set type_sig = gen_type_sig(t) -%}\n  if (args.b_type == {{VLLMScalarTypeTag[t.b]}}\n      && a_type == {{TorchTypeTag[t.a]}}\n      && out_type == {{TorchTypeTag[t.out]}}\n      && {%if t.b_group_scale != void -%}\n      maybe_g_scales_type == {{TorchTypeTag[t.b_group_scale]}}\n      {%- else %}!maybe_g_scales_type{%endif%}\n      && {%if t.b_group_zeropoint != void -%}\n      maybe_g_zeros_type == {{TorchTypeTag[t.b_group_zeropoint]}}\n      {%- else %}!maybe_g_zeros_type{%endif%}\n      && {%if t.b_channel_scale != void -%}\n      maybe_ch_scales_type == {{TorchTypeTag[t.b_channel_scale]}}\n      {%- else %}!maybe_ch_scales_type{%endif%}\n      && {%if t.a_token_scale != void -%}\n      maybe_tok_scales_type == {{TorchTypeTag[t.a_token_scale]}}\n      {%- else %}!maybe_tok_scales_type{%endif%}\n  ) {\n      return mm_dispatch_{{type_sig}}(args);\n  }\n  {%- endfor %}\n  \n  TORCH_CHECK_NOT_IMPLEMENTED(\n    false, "machete_mm(..) is not implemented for "\n    "a_type=", args.A.scalar_type(),\n    ", b_type=", args.b_type.str(),\n    ", out_type=", out_type,\n    ", with_group_scale_type=", maybe_g_scales_type\n        ? toString(*maybe_g_scales_type) : "None",\n    ", with_group_zeropoint_type=", maybe_g_zeros_type\n        ? toString(*maybe_g_zeros_type) : "None",\n    ", with_channel_scale_type=", maybe_ch_scales_type\n        ? toString(*maybe_ch_scales_type) : "None",\n    ", with_token_scale_type=", maybe_tok_scales_type\n        ? toString(*maybe_tok_scales_type) : "None",\n    "; implemented types are: \\n",\n    {%- for impl_config in impl_configs %}\n    {% set t = impl_config.types -%}\n    "\\t{{gen_type_option_name(t)}}\\n",\n    {%- endfor %}\n    "");\n}\n\nstd::vector<std::string> supported_schedules_dispatch(\n    SupportedSchedulesArgs args) {\n    auto out_type = args.maybe_out_type.value_or(args.a_type);\n    \n    {% for impl_config in impl_configs %}\n    {% set t = impl_config.types -%}\n    {% set schs = impl_config.schedules -%}\n    if (args.b_type == {{VLLMScalarTypeTag[t.b]}}\n        && args.a_type == {{TorchTypeTag[t.a]}}\n        && out_type == {{TorchTypeTag[t.out]}}\n        && {%if t.b_group_scale != void -%}\n        args.maybe_group_scales_type == {{TorchTypeTag[t.b_group_scale]}}\n        {%- else %}!args.maybe_group_scales_type{%endif%}\n        && {%if t.b_group_zeropoint != void-%}\n        args.maybe_group_zeros_type == {{TorchTypeTag[t.b_group_zeropoint]}}\n        {%- else %}!args.maybe_group_zeros_type{%endif%}\n    ) {\n        return {\n            {%- for s in impl_config.schedules %}\n            "{{gen_sch_sig(s)}}"{% if not loop.last %},{% endif %}\n            {%- endfor %}\n        };\n    }\n    {%- endfor %}\n    \n    return {};\n};\n\n}; // namespace machete\n'
IMPL_TEMPLATE = '\n#include "../machete_mm_launcher.cuh"\n\nnamespace machete {\n    \n{% for sch in unique_schedules(impl_configs) %}\n{% set sch_sig = gen_sch_sig(sch) -%}\nstruct sch_{{sch_sig}} {\n  using TileShapeNM = Shape<{{\n      to_cute_constant(sch.tile_shape_mn)|join(\', \')}}>;\n  using ClusterShape = Shape<{{\n      to_cute_constant(sch.cluster_shape_mnk)|join(\', \')}}>;\n  // TODO: Reimplement\n  // using KernelSchedule   = {{KernelScheduleTag[sch.kernel_schedule]}};\n  using EpilogueSchedule = {{EpilogueScheduleTag[sch.epilogue_schedule]}};\n  using TileScheduler    = {{TileSchedulerTag[sch.tile_scheduler]}};\n  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;\n};\n{% endfor %}\n    \n{% for impl_config in impl_configs %}\n{% set t = impl_config.types -%}\n{% set schs = impl_config.schedules -%}\n{% set type_sig = gen_type_sig(t) -%}\n\ntemplate<typename Sch>\nusing Kernel_{{type_sig}} = MacheteKernelTemplate<\n  {{DataTypeTag[t.a]}},  // ElementA\n  {{DataTypeTag[t.b]}},  // ElementB\n  {{DataTypeTag[t.out]}},  // ElementD\n  {{DataTypeTag[t.accumulator]}}, // Accumulator\n  {{DataTypeTag[t.b_group_scale]}}, // GroupScaleT\n  {{DataTypeTag[t.b_group_zeropoint]}}, // GroupZeroT\n  {{DataTypeTag[t.b_channel_scale]}}, // ChannelScaleT\n  {{DataTypeTag[t.a_token_scale]}}, // TokenScaleT\n  cutlass::gemm::KernelTmaWarpSpecializedCooperative,\n  Sch>;\n\n{% for sch in schs %}\n{% set sch_sig = gen_sch_sig(sch) -%}\ntorch::Tensor \nimpl_{{type_sig}}_sch_{{sch_sig}}(MMArgs args) {\n  return run_impl<Kernel_{{type_sig}}<sch_{{sch_sig}}>>(args);\n}\n{%- endfor %}\n{%- endfor %}\n\n}; // namespace machete\n'
PREPACK_TEMPLATE = '\n#include "../machete_prepack_launcher.cuh"\n\nnamespace machete {\n\ntorch::Tensor prepack_B_dispatch(PrepackBArgs args) {\n  auto convert_type = args.maybe_group_scales_type.value_or(args.a_type);\n  {%- for t in types %}\n  {% set b_type = unsigned_type_with_bitwidth(t.b_num_bits) %}\n  if (args.a_type == {{TorchTypeTag[t.a]}}\n      && args.b_type.size_bits() == {{t.b_num_bits}} \n      && convert_type == {{TorchTypeTag[t.convert]}}) {\n    return prepack_impl<\n      PrepackedLayoutBTemplate<\n        {{DataTypeTag[t.a]}}, // ElementA\n        {{DataTypeTag[b_type]}}, // ElementB\n        {{DataTypeTag[t.convert]}}, // ElementConvert\n        {{DataTypeTag[t.accumulator]}}, // Accumulator\n        cutlass::layout::ColumnMajor,\n        cutlass::gemm::KernelTmaWarpSpecializedCooperative>\n    >(args.B); \n  }\n  {%- endfor %}\n  \n  TORCH_CHECK_NOT_IMPLEMENTED(false, \n    "prepack_B_dispatch(..) is not implemented for "\n    "atype = ", args.a_type,\n    ", b_type = ", args.b_type.str(),\n    ", with_group_scales_type= ", args.maybe_group_scales_type ? \n        toString(*args.maybe_group_scales_type) : "None");\n}\n\n}; // namespace machete\n'
TmaMI = MixedInputKernelScheduleType.TmaWarpSpecializedCooperative
TmaCoop = EpilogueScheduleType.TmaWarpSpecializedCooperative

@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape_mn: tuple[int, int]
    cluster_shape_mnk: tuple[int, int, int]
    kernel_schedule: MixedInputKernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType

@dataclass(frozen=True)
class TypeConfig:
    a: DataType
    b: Union[DataType, VLLMDataType]
    b_group_scale: DataType
    b_group_zeropoint: DataType
    b_channel_scale: DataType
    a_token_scale: DataType
    out: DataType
    accumulator: DataType

@dataclass(frozen=True)
class PrepackTypeConfig:
    a: DataType
    b_num_bits: int
    convert: DataType
    accumulator: DataType

@dataclass
class ImplConfig:
    types: TypeConfig
    schedules: list[ScheduleConfig]
    heuristic: list[tuple[Optional[str], ScheduleConfig]]

def generate_sch_sig(schedule_config: ScheduleConfig) -> str:
    tile_shape = f'{schedule_config.tile_shape_mn[0]}x{schedule_config.tile_shape_mn[1]}'
    cluster_shape = f'{schedule_config.cluster_shape_mnk[0]}' + f'x{schedule_config.cluster_shape_mnk[1]}' + f'x{schedule_config.cluster_shape_mnk[2]}'
    kernel_schedule = VLLMKernelScheduleTag[schedule_config.kernel_schedule].split('::')[-1]
    epilogue_schedule = EpilogueScheduleTag[schedule_config.epilogue_schedule].split('::')[-1]
    tile_scheduler = TileSchedulerTag[schedule_config.tile_scheduler].split('::')[-1]
    return f'{tile_shape}_{cluster_shape}_{kernel_schedule}' + f'_{epilogue_schedule}_{tile_scheduler}'

def generate_terse_sch_sig(schedule_config: ScheduleConfig) -> str:
    kernel_terse_names_replace = {'KernelTmaWarpSpecializedCooperative': 'TmaMI_', 'TmaWarpSpecializedCooperative_': 'TmaCoop_', 'StreamKScheduler': 'streamK'}
    sch_sig = generate_sch_sig(schedule_config)
    for orig, terse in kernel_terse_names_replace.items():
        sch_sig = sch_sig.replace(orig, terse)
    return sch_sig

def generate_type_signature(kernel_types: TypeConfig):
    return str(''.join([VLLMDataTypeNames[getattr(kernel_types, field.name)] for field in fields(TypeConfig)]))

def generate_type_option_name(kernel_types: TypeConfig):
    return ', '.join([f"{field.name.replace('b_', 'with_') + '_type'}=" + VLLMDataTypeNames[getattr(kernel_types, field.name)] for field in fields(TypeConfig)])

def is_power_of_two(n):
    return n != 0 and n & n - 1 == 0

def to_cute_constant(value: list[int]):

    def _to_cute_constant(value: int):
        if is_power_of_two(value):
            return f'_{value}'
        else:
            return f'Int<{value}>'
    if isinstance(value, Iterable):
        return [_to_cute_constant(value) for value in value]
    else:
        return _to_cute_constant(value)

def unique_schedules(impl_configs: list[ImplConfig]):
    return list(set((sch for impl_config in impl_configs for sch in impl_config.schedules)))

def unsigned_type_with_bitwidth(num_bits):
    return {4: DataType.u4, 8: DataType.u8, 16: DataType.u16, 32: DataType.u32, 64: DataType.u64}[num_bits]
template_globals = {'void': DataType.void, 'DataTypeTag': VLLMDataTypeTag, 'VLLMScalarTypeTag': VLLMDataTypeVLLMScalarTypeTag, 'TorchTypeTag': VLLMDataTypeTorchDataTypeTag, 'KernelScheduleTag': VLLMKernelScheduleTag, 'EpilogueScheduleTag': EpilogueScheduleTag, 'TileSchedulerTag': TileSchedulerTag, 'to_cute_constant': to_cute_constant, 'gen_sch_sig': generate_terse_sch_sig, 'gen_type_sig': generate_type_signature, 'unique_schedules': unique_schedules, 'unsigned_type_with_bitwidth': unsigned_type_with_bitwidth, 'gen_type_option_name': generate_type_option_name}

def create_template(template_str):
    template = jinja2.Template(template_str)
    template.globals.update(template_globals)
    return template
mm_dispatch_template = create_template(DISPATCH_TEMPLATE)
mm_impl_template = create_template(IMPL_TEMPLATE)
prepack_dispatch_template = create_template(PREPACK_TEMPLATE)

def create_sources(impl_configs: list[ImplConfig], num_impl_files=8):
    sources = []
    sources.append(('machete_mm_dispatch', mm_dispatch_template.render(impl_configs=impl_configs)))
    prepack_types = []
    for impl_config in impl_configs:
        convert_type = impl_config.types.a if impl_config.types.b_group_scale == DataType.void else impl_config.types.b_group_scale
        prepack_types.append(PrepackTypeConfig(a=impl_config.types.a, b_num_bits=VLLMDataTypeSize[impl_config.types.b], convert=convert_type, accumulator=impl_config.types.accumulator))

    def prepacked_type_key(prepack_type: PrepackTypeConfig):
        return (prepack_type.a, prepack_type.b_num_bits, prepack_type.convert)
    unique_prepack_types = []
    prepack_types_seen = set()
    for prepack_type in prepack_types:
        key = prepacked_type_key(prepack_type)
        if key not in prepack_types_seen:
            unique_prepack_types.append(prepack_type)
            prepack_types_seen.add(key)
    sources.append(('machete_prepack', prepack_dispatch_template.render(types=unique_prepack_types)))
    num_impls = reduce(lambda x, y: x + len(y.schedules), impl_configs, 0)
    num_impls_per_file = math.ceil(num_impls / num_impl_files)
    files_impls: list[list[ImplConfig]] = [[]]
    curr_num_impls_assigned = 0
    curr_impl_in_file = 0
    curr_impl_configs = deepcopy(list(reversed(impl_configs)))
    while curr_num_impls_assigned < num_impls:
        room_left_in_file = num_impls_per_file - curr_impl_in_file
        if room_left_in_file == 0:
            files_impls.append([])
            room_left_in_file = num_impls_per_file
            curr_impl_in_file = 0
        curr_ic = curr_impl_configs[-1]
        if len(curr_ic.schedules) >= room_left_in_file:
            tmp_ic = deepcopy(curr_ic)
            tmp_ic.schedules = curr_ic.schedules[:room_left_in_file]
            curr_ic.schedules = curr_ic.schedules[room_left_in_file:]
            files_impls[-1].append(tmp_ic)
        else:
            files_impls[-1].append(curr_ic)
            curr_impl_configs.pop()
        curr_num_impls_assigned += len(files_impls[-1][-1].schedules)
        curr_impl_in_file += len(files_impls[-1][-1].schedules)
    for part, file_impls in enumerate(files_impls):
        sources.append((f'machete_mm_impl_part{part + 1}', mm_impl_template.render(impl_configs=file_impls)))
    return sources

def generate():
    SCRIPT_DIR = os.path.dirname(__file__)
    sch_common_params = dict(kernel_schedule=TmaMI, epilogue_schedule=TmaCoop, tile_scheduler=TileSchedulerType.StreamK)
    default_tile_heuristic_config = {'M > 256 && K <= 16384 && N <= 4096': ((128, 128), (2, 1, 1)), 'M > 256': ((128, 256), (2, 1, 1)), 'M > 128 && K <= 4096 && N <= 4096': ((128, 64), (2, 1, 1)), 'M > 128 && K <= 8192 && N <= 8192': ((128, 128), (2, 1, 1)), 'M > 128': ((128, 256), (2, 1, 1)), 'M > 64 && K <= 4069 && N <= 4069': ((128, 32), (2, 1, 1)), 'M > 64 && K <= 4069 && N <= 8192': ((128, 64), (2, 1, 1)), 'M > 64 && K >= 8192 && N >= 12288': ((256, 128), (2, 1, 1)), 'M > 64': ((128, 128), (2, 1, 1)), 'M > 32 && K <= 6144 && N <= 6144': ((128, 16), (1, 1, 1)), 'M > 32 && K >= 16384 && N >= 12288': ((256, 64), (2, 1, 1)), 'M > 32': ((128, 64), (2, 1, 1)), 'M > 16 && K <= 12288 && N <= 8192': ((128, 32), (2, 1, 1)), 'M > 16': ((256, 32), (2, 1, 1)), 'N >= 26624': ((256, 16), (1, 1, 1)), None: ((128, 16), (1, 1, 1))}
    default_heuristic = [(cond, ScheduleConfig(*tile_config, **sch_common_params)) for cond, tile_config in default_tile_heuristic_config.items()]

    def get_unique_schedules(heuristic: dict[str, ScheduleConfig]):
        schedules = []
        for _, schedule_config in heuristic:
            if schedule_config not in schedules:
                schedules.append(schedule_config)
        return schedules
    impl_configs = []
    GPTQ_kernel_type_configs = list((TypeConfig(a=a, b=b, b_group_scale=a, b_group_zeropoint=DataType.void, b_channel_scale=DataType.void, a_token_scale=DataType.void, out=a, accumulator=DataType.f32) for b in (VLLMDataType.u4b8, VLLMDataType.u8b128) for a in (DataType.f16, DataType.bf16)))
    impl_configs += [ImplConfig(x[0], x[1], x[2]) for x in zip(GPTQ_kernel_type_configs, itertools.repeat(get_unique_schedules(default_heuristic)), itertools.repeat(default_heuristic))]
    AWQ_kernel_type_configs = list((TypeConfig(a=a, b=b, b_group_scale=a, b_group_zeropoint=a, b_channel_scale=DataType.void, a_token_scale=DataType.void, out=a, accumulator=DataType.f32) for b in (DataType.u4, DataType.u8) for a in (DataType.f16, DataType.bf16)))
    impl_configs += [ImplConfig(x[0], x[1], x[2]) for x in zip(AWQ_kernel_type_configs, itertools.repeat(get_unique_schedules(default_heuristic)), itertools.repeat(default_heuristic))]
    qqq_tile_heuristic_config = {'M > 256': ((128, 128), (2, 1, 1)), 'M > 128 && K <= 4096 && N <= 4096': ((128, 64), (2, 1, 1)), 'M > 128 && K <= 8192 && N <= 8192': ((128, 128), (2, 1, 1)), 'M > 128': ((128, 128), (2, 1, 1)), 'M > 64 && K <= 4069 && N <= 4069': ((128, 32), (2, 1, 1)), 'M > 64 && K <= 4069 && N <= 8192': ((128, 64), (2, 1, 1)), 'M > 64 && K >= 8192 && N >= 12288': ((256, 128), (2, 1, 1)), 'M > 64': ((128, 128), (2, 1, 1)), 'M > 32 && K <= 6144 && N <= 6144': ((128, 16), (1, 1, 1)), 'M > 32': ((128, 64), (2, 1, 1)), 'M > 16 && K <= 12288 && N <= 8192': ((128, 32), (2, 1, 1)), 'M > 16': ((256, 32), (2, 1, 1)), 'N >= 26624': ((256, 16), (1, 1, 1)), None: ((128, 16), (1, 1, 1))}
    qqq_heuristic = [(cond, ScheduleConfig(*tile_config, **sch_common_params)) for cond, tile_config in qqq_tile_heuristic_config.items()]
    QQQ_kernel_types = [*(TypeConfig(a=DataType.s8, b=VLLMDataType.u4b8, b_group_scale=b_group_scale, b_group_zeropoint=DataType.void, b_channel_scale=DataType.f32, a_token_scale=DataType.f32, out=DataType.f16, accumulator=DataType.s32) for b_group_scale in (DataType.f16, DataType.void)), *(TypeConfig(a=DataType.e4m3, b=VLLMDataType.u4b8, b_group_scale=b_group_scale, b_group_zeropoint=DataType.void, b_channel_scale=DataType.f32, a_token_scale=DataType.f32, out=DataType.f16, accumulator=DataType.f32) for b_group_scale in (DataType.f16, DataType.void))]
    impl_configs += [ImplConfig(x[0], x[1], x[2]) for x in zip(QQQ_kernel_types, itertools.repeat(get_unique_schedules(qqq_heuristic)), itertools.repeat(qqq_heuristic))]
    output_dir = os.path.join(SCRIPT_DIR, 'generated')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for filename, code in create_sources(impl_configs):
        filepath = os.path.join(output_dir, f'{filename}.cu')
        with open(filepath, 'w') as output_file:
            output_file.write(code)
        print(f'Rendered template to {filepath}')
if __name__ == '__main__':
    generate()