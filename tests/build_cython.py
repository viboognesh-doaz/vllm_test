from Cython.Build import cythonize
from setuptools import setup
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
infiles = []
infiles += ['vllm/engine/llm_engine.py', 'vllm/transformers_utils/detokenizer.py', 'vllm/engine/output_processor/single_step.py', 'vllm/outputs.py', 'vllm/engine/output_processor/stop_checker.py']
infiles += ['vllm/core/scheduler.py', 'vllm/sequence.py', 'vllm/core/block_manager.py']
infiles += ['vllm/model_executor/layers/sampler.py', 'vllm/sampling_params.py', 'vllm/utils/__init__.py']
setup(ext_modules=cythonize(infiles, annotate=False, force=True, compiler_directives={'language_level': '3', 'infer_types': True}))