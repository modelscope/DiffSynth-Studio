from __future__ import annotations

import ast
import json
import re
from pathlib import Path


_OUTPUT_POSITIONAL_ARGUMENTS = {
    "save": 0,
    "save_audio": 2,
    "save_video": 1,
    "export_to_video": 1,
    "write_video_audio": 2,
}


def _function_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


class _SamplingArguments(ast.NodeTransformer):
    def __init__(self, prompt: str, output: Path) -> None:
        self.prompt = prompt
        self.output = str(output)
        self.prompt_count = 0
        self.output_count = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets):
            node.value = ast.Constant(value=self.prompt)
            self.prompt_count += 1
        if any(isinstance(target, ast.Name) and target.id == "output_path" for target in node.targets):
            node.value = ast.Constant(value=self.output)
            self.output_count += 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        for keyword in node.keywords:
            if keyword.arg == "prompt" and isinstance(keyword.value, ast.Constant):
                keyword.value = ast.Constant(value=self.prompt)
                self.prompt_count += 1
            elif keyword.arg == "output_path":
                keyword.value = ast.Constant(value=self.output)
                self.output_count += 1

        output_argument = _OUTPUT_POSITIONAL_ARGUMENTS.get(_function_name(node))
        if output_argument is not None and len(node.args) > output_argument:
            node.args[output_argument] = ast.Constant(value=self.output)
            self.output_count += 1
        return node


def adapted_source(script: Path, checkpoint: Path, prompt: str, output: Path) -> str:
    source = script.read_text(encoding="utf-8")
    source = re.sub(r"(?m)^(\s*)#\s*(pipe\.load_lora\()", r"\1\2", source)
    checkpoint_literal = json.dumps(str(checkpoint))
    source, direct_count = re.subn(
        r"(pipe\.load_lora\([^,\n]+,\s*)([\"']).*?\2",
        lambda match: f"{match.group(1)}{checkpoint_literal}",
        source,
    )
    state_dict_names = set(
        re.findall(r"pipe\.load_lora\([^\n]*?\bstate_dict\s*=\s*([A-Za-z_]\w*)", source)
    )
    state_dict_count = 0
    for name in state_dict_names:
        pattern = (
            rf"(?m)^(\s*{re.escape(name)}\s*=\s*load_state_dict\(\s*)"
            rf"(?:[rRuUbBfF]*)([\"']).*?\2"
        )
        source, count = re.subn(
            pattern,
            lambda match: f"{match.group(1)}{checkpoint_literal}",
            source,
            count=1,
        )
        state_dict_count += count
    if direct_count + state_dict_count == 0:
        raise ValueError(f"cannot locate LoRA checkpoint in {script}")

    tree = ast.parse(source, filename=str(script))
    adapter = _SamplingArguments(prompt, output)
    tree = adapter.visit(tree)
    ast.fix_missing_locations(tree)
    if adapter.prompt_count == 0:
        raise ValueError(f"cannot locate prompt in {script}")
    if adapter.output_count == 0:
        raise ValueError(f"cannot locate sample output in {script}")

    return ast.unparse(tree) + "\n"
