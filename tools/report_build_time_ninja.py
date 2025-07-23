from collections import defaultdict
import argparse
import errno
import fnmatch
import os
import sys
"Summarize the last ninja build, invoked with ninja's -C syntax.\n\n> python3 tools/report_build_time_ninja.py -C build/..\n\nTypical output looks like this:\n```\n    Longest build steps for .cpp.o:\n           1.0 weighted s to build ...torch_bindings.cpp.o (12.4 s elapsed time)\n           2.0 weighted s to build ..._attn_c.dir/csrc... (23.5 s elapsed time)\n           2.6 weighted s to build ...torch_bindings.cpp.o (31.5 s elapsed time)\n           3.2 weighted s to build ...torch_bindings.cpp.o (38.5 s elapsed time)\n    Longest build steps for .so (linking):\n           0.1 weighted s to build _moe_C.abi3.so (1.0 s elapsed time)\n           0.5 weighted s to build ...flash_attn_c.abi3.so (1.1 s elapsed time)\n           6.2 weighted s to build _C.abi3.so (6.2 s elapsed time)\n    Longest build steps for .cu.o:\n          15.3 weighted s to build ...machete_mm_... (183.5 s elapsed time)\n          15.3 weighted s to build ...machete_mm_... (183.5 s elapsed time)\n          15.3 weighted s to build ...machete_mm_... (183.6 s elapsed time)\n          15.3 weighted s to build ...machete_mm_... (183.7 s elapsed time)\n          15.5 weighted s to build ...machete_mm_... (185.6 s elapsed time)\n          15.5 weighted s to build ...machete_mm_... (185.9 s elapsed time)\n          15.5 weighted s to build ...machete_mm_... (186.2 s elapsed time)\n          37.4 weighted s to build ...scaled_mm_c3x.cu... (449.0 s elapsed time)\n          43.9 weighted s to build ...scaled_mm_c2x.cu... (527.4 s elapsed time)\n         344.8 weighted s to build ...attention_...cu.o (1087.2 s elapsed time)\n    1110.0 s weighted time (10120.4 s elapsed time sum, 9.1x parallelism)\n    134 build steps completed, average of 0.12/s\n```\n"
long_count = 10
long_ext_count = 10

class Target:
    """Represents a single line read for a .ninja_log file."""

    def __init__(self, start, end):
        """Creates a target object by passing in the start/end times in seconds
        as a float."""
        self.start = start
        self.end = end
        self.targets = []
        self.weighted_duration = 0.0

    def Duration(self):
        """Returns the task duration in seconds as a float."""
        return self.end - self.start

    def SetWeightedDuration(self, weighted_duration):
        """Sets the duration, in seconds, passed in as a float."""
        self.weighted_duration = weighted_duration

    def WeightedDuration(self):
        """Returns the task's weighted duration in seconds as a float.

        Weighted_duration takes the elapsed time of the task and divides it
        by how many other tasks were running at the same time. Thus, it
        represents the approximate impact of this task on the total build time,
        with serialized or serializing steps typically ending up with much
        longer weighted durations.
        weighted_duration should always be the same or shorter than duration.
        """
        epsilon = 2e-06
        if self.weighted_duration > self.Duration() + epsilon:
            print('{} > {}?'.format(self.weighted_duration, self.Duration()))
        assert self.weighted_duration <= self.Duration() + epsilon
        return self.weighted_duration

    def DescribeTargets(self):
        """Returns a printable string that summarizes the targets."""
        result = ', '.join(self.targets)
        max_length = 65
        if len(result) > max_length:
            result = result[:max_length] + '...'
        return result

def ReadTargets(log, show_all):
    """Reads all targets from .ninja_log file |log_file|, sorted by duration.

    The result is a list of Target objects."""
    header = log.readline()
    assert header == '# ninja log v5\n', 'unrecognized ninja log version {!r}'.format(header)
    targets_dict = {}
    last_end_seen = 0.0
    for line in log:
        parts = line.strip().split('\t')
        if len(parts) != 5:
            continue
        start, end, _, name, cmdhash = parts
        start = int(start) / 1000.0
        end = int(end) / 1000.0
        if not show_all and end < last_end_seen:
            targets_dict = {}
        target = None
        if cmdhash in targets_dict:
            target = targets_dict[cmdhash]
            if not show_all and (target.start != start or target.end != end):
                targets_dict = {}
                target = None
        if not target:
            targets_dict[cmdhash] = target = Target(start, end)
        last_end_seen = end
        target.targets.append(name)
    return list(targets_dict.values())

def GetExtension(target, extra_patterns):
    """Return the file extension that best represents a target.

  For targets that generate multiple outputs it is important to return a
  consistent 'canonical' extension. Ultimately the goal is to group build steps
  by type."""
    for output in target.targets:
        if extra_patterns:
            for fn_pattern in extra_patterns.split(';'):
                if fnmatch.fnmatch(output, '*' + fn_pattern + '*'):
                    return fn_pattern
        if output.endswith('type_mappings'):
            extension = 'type_mappings'
            break
        root, ext1 = os.path.splitext(output)
        _, ext2 = os.path.splitext(root)
        extension = ext2 + ext1
        if len(extension) == 0:
            extension = '(no extension found)'
        if ext1 in ['.pdb', '.dll', '.exe']:
            extension = 'PEFile (linking)'
            break
        if ext1 in ['.so', '.TOC']:
            extension = '.so (linking)'
            break
        if ext1 in ['.obj', '.o']:
            break
        if ext1 == '.jar':
            break
        if output.count('.mojom') > 0:
            extension = 'mojo'
            break
    return extension

def SummarizeEntries(entries, extra_step_types):
    """Print a summary of the passed in list of Target objects."""
    task_start_stop_times = []
    earliest = -1
    latest = 0
    total_cpu_time = 0
    for target in entries:
        if earliest < 0 or target.start < earliest:
            earliest = target.start
        if target.end > latest:
            latest = target.end
        total_cpu_time += target.Duration()
        task_start_stop_times.append((target.start, 'start', target))
        task_start_stop_times.append((target.end, 'stop', target))
    length = latest - earliest
    weighted_total = 0.0
    task_start_stop_times.sort(key=lambda times: times[:2])
    running_tasks = {}
    last_time = task_start_stop_times[0][0]
    last_weighted_time = 0.0
    for event in task_start_stop_times:
        time, action_name, target = event
        num_running = len(running_tasks)
        if num_running > 0:
            last_weighted_time += (time - last_time) / float(num_running)
        if action_name == 'start':
            running_tasks[target] = last_weighted_time
        if action_name == 'stop':
            weighted_duration = last_weighted_time - running_tasks[target]
            target.SetWeightedDuration(weighted_duration)
            weighted_total += weighted_duration
            del running_tasks[target]
        last_time = time
    assert len(running_tasks) == 0
    if abs(length - weighted_total) > 500:
        print('Warning: Possible corrupt ninja log, results may be untrustworthy. Length = {:.3f}, weighted total = {:.3f}'.format(length, weighted_total))
    entries_by_ext = defaultdict(list)
    for target in entries:
        extension = GetExtension(target, extra_step_types)
        entries_by_ext[extension].append(target)
    for key, values in entries_by_ext.items():
        print('    Longest build steps for {}:'.format(key))
        values.sort(key=lambda x: x.WeightedDuration())
        for target in values[-long_count:]:
            print('      {:8.1f} weighted s to build {} ({:.1f} s elapsed time)'.format(target.WeightedDuration(), target.DescribeTargets(), target.Duration()))
    print('    {:.1f} s weighted time ({:.1f} s elapsed time sum, {:1.1f}x parallelism)'.format(length, total_cpu_time, total_cpu_time * 1.0 / length))
    print('    {} build steps completed, average of {:1.2f}/s'.format(len(entries), len(entries) / length))

def main():
    log_file = '.ninja_log'
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', dest='build_directory', help='Build directory.')
    parser.add_argument('-s', '--step-types', help='semicolon separated fnmatch patterns for build-step grouping')
    parser.add_argument('--log-file', help='specific ninja log file to analyze.')
    args, _extra_args = parser.parse_known_args()
    if args.build_directory:
        log_file = os.path.join(args.build_directory, log_file)
    if args.log_file:
        log_file = args.log_file
    if args.step_types:
        global long_ext_count
        long_ext_count += len(args.step_types.split(';'))
    try:
        with open(log_file) as log:
            entries = ReadTargets(log, False)
            SummarizeEntries(entries, args.step_types)
    except OSError:
        print('Log file {!r} not found, no build summary created.'.format(log_file))
        return errno.ENOENT
if __name__ == '__main__':
    sys.exit(main())