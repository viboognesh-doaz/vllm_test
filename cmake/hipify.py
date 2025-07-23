from torch.utils.hipify.hipify_python import hipify
import argparse
import os
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_dir', help='The project directory.')
    parser.add_argument('-o', '--output_dir', help='The output directory.')
    parser.add_argument('sources', help='Source files to hipify.', nargs='*', default=[])
    args = parser.parse_args()
    includes = [os.path.join(args.project_dir, '*')]
    extra_files = [os.path.abspath(s) for s in args.sources]
    shutil.copytree(args.project_dir, args.output_dir, dirs_exist_ok=True)
    hipify_result = hipify(project_directory=args.project_dir, output_directory=args.output_dir, header_include_dirs=[], includes=includes, extra_files=extra_files, show_detailed=True, is_pytorch_extension=True, hipify_extra_files_only=True)
    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = hipify_result[s_abs].hipified_path if s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None else s_abs
        hipified_sources.append(hipified_s_abs)
    assert len(hipified_sources) == len(args.sources)
    print('\n'.join(hipified_sources))