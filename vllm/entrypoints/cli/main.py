from __future__ import annotations
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
from vllm.utils import FlexibleArgumentParser
import importlib.metadata
import vllm.entrypoints.cli.benchmark.main
import vllm.entrypoints.cli.collect_env
import vllm.entrypoints.cli.openai
import vllm.entrypoints.cli.run_batch
import vllm.entrypoints.cli.serve
'The CLI entrypoints of vLLM\n\nNote that all future modules must be lazily loaded within main\nto avoid certain eager import breakage.'

def main():
    CMD_MODULES = [vllm.entrypoints.cli.openai, vllm.entrypoints.cli.serve, vllm.entrypoints.cli.benchmark.main, vllm.entrypoints.cli.collect_env, vllm.entrypoints.cli.run_batch]
    cli_env_setup()
    parser = FlexibleArgumentParser(description='vLLM CLI', epilog=VLLM_SUBCMD_PARSER_EPILOG)
    parser.add_argument('-v', '--version', action='version', version=importlib.metadata.version('vllm'))
    subparsers = parser.add_subparsers(required=False, dest='subparser')
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)
    if hasattr(args, 'dispatch_function'):
        args.dispatch_function(args)
    else:
        parser.print_help()
if __name__ == '__main__':
    main()