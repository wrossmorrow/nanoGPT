import argparse

from dataclasses import dataclass, fields, Field, _MISSING_TYPE
from typing import Any, Dict, get_type_hints, List, Optional, Tuple

from torch.cuda import is_available as cuda_is_available

from . import config


def _add_field(parser: argparse.ArgumentParser, field: Field) -> None:
    field_flag = "--" + field.name.replace("_", "-")
    if isinstance(field.default, _MISSING_TYPE):
        parser.add_argument(
            field_flag,
            type=field.type,
            required=True,  # problematic if we allow config files
            help=field.metadata.get("help", "-"),
        )
    else:
        parser.add_argument(
            field_flag,
            type=field.type,
            default=field.default,
            help=field.metadata.get("help", "-"),
        )


@dataclass
class CLICommand:
    name: str
    conf_classes: List[Any]

    def parse_config(self, args: argparse.Namespace) -> List[Any]:
        confs = []
        for cc in self.conf_classes:
            conf_args = {
                fld: getattr(args, fld)
                for fld in [f.name for f in fields(cc) if f.init and f.metadata.get("cli", True)]
            }
            conf = cc.from_file(args.config_file, **conf_args) if args.config_file else cc(**conf_args)
            confs.append(conf)
        return confs


@dataclass
class CLIArgument:
    name: str
    type: Any
    required: bool = False
    default: Optional[Any] = None
    help: str = "-"

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        flag = f"--{self.name.lower().replace('_', '-')}"
        if self.required:
            parser.add_argument(
                flag,
                type=self.type,
                required=True,
                help=self.help,
            )
        else:
            parser.add_argument(
                flag,
                type=self.type,
                default=self.default,
                help=self.help,
            )


class CLIParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog="nanoGPT (forked)", description="Utility for running GPT-like models", epilog="Not for commercial use"
        )
        self.subparsers = self.parser.add_subparsers(dest="command", help="sub-command help")
        self.commands: Dict[str, CLICommand] = {}

    def add_cmd(
        self,
        name: str,
        conf_classes: List[Any],
        config_file: bool = True,
        extra_args: Optional[List[CLIArgument]] = None,
    ) -> argparse.ArgumentParser:
        self.commands[name] = CLICommand(name=name, conf_classes=conf_classes)
        parser = self.subparsers.add_parser(name)
        if config_file:
            parser.add_argument(
                "-f",
                "--config-file",
                type=str,
                required=False,
                help="Configuration file",
            )
        for cc in conf_classes:
            resolved = get_type_hints(cc)  # dataclass.Field types are strings...
            for fld in fields(cc):
                if fld.init and fld.metadata.get("cli", True):
                    fld.type = resolved[fld.name]
                    _add_field(parser, fld)
        if extra_args:
            for arg in extra_args:
                arg.add_to_parser(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        assert args.command is not None, "Command required"
        return args

    def parse_config(self) -> Tuple[str, Dict[str, Any], argparse.Namespace]:
        args = self.parse_args()
        configs = self.commands[args.command].parse_config(args)
        return args.command, {conf.__class__.__name__: conf for conf in configs}, args


default_cmd_configs = {
    "eval": [config.DatasetConfig, config.CheckpointConfig, config.EvaluateConfig],
    "train": [config.DatasetConfig, config.NanoGPTConfig, config.TrainingConfig],
    "resume": [config.DatasetConfig, config.CheckpointConfig, config.TrainingConfig],
    "generate": [config.DatasetConfig, config.GenerateConfig],
}

DeviceCLIArg = CLIArgument(
    name="device",
    type=str,
    # choices=["cpu", "cuda"],  # ...
    default="cuda" if cuda_is_available() else "cpu",
    help="Device to use for model evaluation or training",
)

DefaultCLI = CLIParser()
for cmd, confs in default_cmd_configs.items():
    DefaultCLI.add_cmd(cmd, confs, [DeviceCLIArg])
