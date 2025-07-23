from vllm.benchmarks.serve import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
import argparse

class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """ The `serve` subcommand for vllm bench. """
    name = 'serve'
    help = 'Benchmark the online serving throughput.'

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)