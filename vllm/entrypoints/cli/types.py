from __future__ import annotations
from vllm.utils import FlexibleArgumentParser
import argparse
import typing
if typing.TYPE_CHECKING:

class CLISubcommand:
    """Base class for CLI argument handlers."""
    name: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError('Subclasses should implement this method')

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        raise NotImplementedError('Subclasses should implement this method')