from contextlib import nullcontext
from vllm_test_utils import BlameResult, blame
import sys
import vllm
module_names = ['torch._inductor.async_compile', 'cv2']

def any_module_imported():
    return any((module_name in sys.modules for module_name in module_names))
use_blame = False
context = blame(any_module_imported) if use_blame else nullcontext()
with context as result:
if use_blame:
    assert isinstance(result, BlameResult)
    print(f'the first import location is:\n{result.trace_stack}')
assert not any_module_imported(), f'Some the modules in {module_names} are imported. To see the first import location, run the test with `use_blame=True`.'