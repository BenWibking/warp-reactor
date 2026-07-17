#!/usr/bin/env python3
"""Parse and partition the generated primordial chemistry DAGs.

The emitted Mojo is deliberately ordinary source: each warp function contains
the transitive closure of its outputs in original topological order.  The
report separately identifies profitable exchange candidates for the GPU
shared-memory implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TEMP_RE = re.compile(r"\bx[0-9]+_[0-9]+\b")
DEF_RE = re.compile(r"^var\s+(x[0-9]+_[0-9]+)\s*=", re.DOTALL)
FUNCTION_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MANIFEST = {
    "rhs_specie": (119, 14),
    "rhs_eint": (162, 1),
    "jac_nuc": (1394, 225),
}


class DagError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Statement:
    text: str
    line: int


@dataclasses.dataclass(frozen=True)
class Definition:
    name: str
    statement: Statement
    dependencies: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Output:
    key: tuple[int, ...]
    statement: Statement
    dependencies: tuple[str, ...]


@dataclasses.dataclass
class FunctionDag:
    name: str
    prelude: tuple[Statement, ...]
    definitions: tuple[Definition, ...]
    outputs: tuple[Output, ...]

    @property
    def by_name(self) -> dict[str, Definition]:
        return {definition.name: definition for definition in self.definitions}


def _delimiter_delta(line: str) -> int:
    depth = 0
    quote = ""
    escaped = False
    for char in line:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def function_body(source: str, name: str) -> tuple[list[str], int]:
    lines = source.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = FUNCTION_RE.match(line)
        if match and match.group(1) == name:
            start = index + 1
            break
    if start is None:
        raise DagError(f"missing function {name}")
    end = start
    while end < len(lines):
        line = lines[end]
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        end += 1
    return lines[start:end], start + 1


def collect_statements(body: list[str], first_line: int) -> list[Statement]:
    statements: list[Statement] = []
    current: list[str] = []
    current_line = first_line
    depth = 0
    for offset, raw in enumerate(body):
        if not raw.strip():
            continue
        if not current:
            if not raw.startswith("    "):
                raise DagError(f"line {first_line + offset}: malformed indentation")
            current_line = first_line + offset
        current.append(raw[4:] if len(current) == 0 else raw)
        depth += _delimiter_delta(raw)
        if depth < 0:
            raise DagError(f"line {first_line + offset}: unmatched closing delimiter")
        if depth == 0:
            statements.append(Statement("\n".join(current), current_line))
            current = []
    if current:
        raise DagError(f"line {current_line}: unterminated statement")
    return statements


def _dependencies(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(TEMP_RE.findall(text)))


def _parse_output(function: str, statement: Statement) -> Output | None:
    text = statement.text.strip()
    if function == "rhs_specie" and text.startswith("vset(ydot,"):
        match = re.match(r"vset\(ydot,\s*(\d+)\s*,", text)
        if not match:
            raise DagError(f"line {statement.line}: malformed ydot output")
        return Output((int(match.group(1)),), statement, _dependencies(text))
    if function == "rhs_eint" and text.startswith("return "):
        return Output((14,), statement, _dependencies(text))
    if function == "jac_nuc" and text.startswith("mset(jac,"):
        match = re.match(r"mset\(jac,\s*(\d+)\s*,\s*(\d+)\s*,", text)
        if not match:
            raise DagError(f"line {statement.line}: malformed Jacobian output")
        return Output(
            (int(match.group(1)), int(match.group(2))),
            statement,
            _dependencies(text),
        )
    return None


def parse_function(source: str, name: str, *, pin_manifest: bool = True) -> FunctionDag:
    body, first_line = function_body(source, name)
    statements = collect_statements(body, first_line)
    definitions: list[Definition] = []
    outputs: list[Output] = []
    prelude: list[Statement] = []
    seen: set[str] = set()

    for statement in statements:
        text = statement.text.strip()
        definition_match = DEF_RE.match(text)
        if definition_match:
            name_token = definition_match.group(1)
            if name_token in seen:
                raise DagError(f"line {statement.line}: duplicate definition {name_token}")
            expression = text[definition_match.end() :]
            dependencies = _dependencies(expression)
            for dependency in dependencies:
                if dependency not in seen:
                    raise DagError(
                        f"line {statement.line}: unresolved or use-before-definition "
                        f"{dependency}"
                    )
            definitions.append(Definition(name_token, statement, dependencies))
            seen.add(name_token)
            continue

        output = _parse_output(name, statement)
        if output is not None:
            for dependency in output.dependencies:
                if dependency not in seen:
                    raise DagError(
                        f"line {statement.line}: unresolved output dependency {dependency}"
                    )
            outputs.append(output)
            continue

        if not definitions and text == "var T = state.T":
            prelude.append(statement)
            continue
        raise DagError(f"line {statement.line}: unsupported statement in {name}: {text[:60]}")

    dag = FunctionDag(name, tuple(prelude), tuple(definitions), tuple(outputs))
    if pin_manifest:
        expected = MANIFEST[name]
        actual = (len(definitions), len(outputs))
        if actual != expected:
            raise DagError(
                f"{name} manifest drift: expected {expected[0]}/{expected[1]}, "
                f"found {actual[0]}/{actual[1]}"
            )
    return dag


def transitive_closure(dag: FunctionDag, output: Output) -> set[str]:
    by_name = dag.by_name
    closure: set[str] = set()
    pending = list(output.dependencies)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        pending.extend(by_name[name].dependencies)
    return closure


def assign_outputs(dag: FunctionDag, warps: int) -> list[list[Output]]:
    assigned: list[list[Output]] = [[] for _ in range(warps)]
    for output in dag.outputs:
        if dag.name == "jac_nuc":
            warp = min(warps - 1, output.key[0] * warps // 15)
        else:
            warp = output.key[0] % warps
        assigned[warp].append(output)
    return assigned


def operation_cost(statement: str) -> int:
    expensive = 0
    for call, weight in (("exp(", 25), ("log(", 25), ("sqrt(", 20), ("cbrt(", 20)):
        expensive += statement.count(call) * weight
    arithmetic = sum(statement.count(operator) for operator in ("+", "-", "*", "/"))
    return max(1, expensive + arithmetic)


def subtree_costs(dag: FunctionDag) -> dict[str, int]:
    costs: dict[str, int] = {}
    for definition in dag.definitions:
        costs[definition.name] = operation_cost(definition.statement.text) + sum(
            costs[dependency] for dependency in definition.dependencies
        )
    return costs


@dataclasses.dataclass
class Partition:
    outputs: list[list[Output]]
    closures: list[set[str]]
    exchanges: set[str]
    exchange_bytes: int


def partition(dag: FunctionDag, warps: int, threshold: int) -> Partition:
    outputs = assign_outputs(dag, warps)
    closures: list[set[str]] = []
    claims: dict[str, int] = defaultdict(int)
    for warp_outputs in outputs:
        closure: set[str] = set()
        for output in warp_outputs:
            closure.update(transitive_closure(dag, output))
        closures.append(closure)
        for name in closure:
            claims[name] += 1
    costs = subtree_costs(dag)
    exchanges = {
        name
        for name, count in claims.items()
        if count >= 2 and costs[name] >= threshold
    }
    # One Float64 per batch cell for each exchanged scalar.
    exchange_bytes = len(exchanges) * 8 * warps
    return Partition(outputs, closures, exchanges, exchange_bytes)


def _indent_statement(text: str) -> str:
    return "\n".join("    " + line.lstrip() if index == 0 else line for index, line in enumerate(text.splitlines()))


def emit_function(
    dag: FunctionDag,
    warp: int,
    closure: set[str],
    outputs: Iterable[Output],
) -> str:
    if dag.name == "jac_nuc":
        signature = (
            f"def jac_slice_{warp}(state: BurnState, mut jac: MatTensor, "
            "X: SpeciesTensor, z: Float64):"
        )
    elif dag.name == "rhs_specie":
        signature = (
            f"def rhs_specie_slice_{warp}(state: BurnState, mut ydot: VecTensor, "
            "X: SpeciesTensor, z: Float64):"
        )
    else:
        signature = (
            f"def rhs_eint_slice_{warp}(state: BurnState, X: SpeciesTensor, "
            "z: Float64) -> Float64:"
        )
    lines = [signature]
    for statement in dag.prelude:
        lines.append(_indent_statement(statement.text))
    for definition in dag.definitions:
        if definition.name in closure:
            lines.append(_indent_statement(definition.statement.text))
    output_list = list(outputs)
    if output_list:
        lines.extend(_indent_statement(output.statement.text) for output in output_list)
    elif dag.name == "rhs_eint":
        lines.append("    return 0.0")
    else:
        lines.append("    pass")
    return "\n".join(lines)


def emit_file(
    path: Path,
    source_hash: str,
    dags: list[FunctionDag],
    partitions: list[Partition],
    warps: int,
    threshold: int,
) -> None:
    header = (
        "# Generated by tools/slice_dag.py; do not edit.\n"
        f"# input_sha256={source_hash} warps={warps} threshold={threshold}\n\n"
        "from std.math import abs, cbrt, exp, log\n"
        "from reproducer import BurnState, MatTensor, SpeciesTensor, VecTensor\n"
        "from reproducer import Log10, Pi, mset, portable_mul, powi_m3, powi_m5, powi_m7\n"
        "from reproducer import portable_sqrt, portable_sqrt as sqrt, truthy, vget, vset\n\n"
    )
    functions: list[str] = []
    for dag, dag_partition in zip(dags, partitions):
        for warp in range(warps):
            functions.append(
                emit_function(
                    dag,
                    warp,
                    dag_partition.closures[warp],
                    dag_partition.outputs[warp],
                )
            )
    path.write_text(header + "\n\n".join(functions) + "\n")


def make_report(
    source_hash: str,
    dags: list[FunctionDag],
    partitions: list[Partition],
    warps: int,
    threshold: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "input_sha256": source_hash,
        "warps": warps,
        "threshold": threshold,
        "functions": {},
    }
    function_reports: dict[str, object] = {}
    for dag, dag_partition in zip(dags, partitions):
        costs = subtree_costs(dag)
        warp_reports = []
        for warp in range(warps):
            closure = dag_partition.closures[warp]
            warp_reports.append(
                {
                    "warp": warp,
                    "outputs": [list(output.key) for output in dag_partition.outputs[warp]],
                    "definitions": len(closure),
                    "weighted_operations": sum(costs[name] for name in closure),
                    "estimated_peak_live_temporaries": len(closure),
                }
            )
        function_reports[dag.name] = {
            "definitions": len(dag.definitions),
            "outputs": len(dag.outputs),
            "exchanged_nodes": sorted(dag_partition.exchanges),
            "recomputed_nodes": sorted(
                set().union(*dag_partition.closures) - dag_partition.exchanges
            ),
            "exchange_bytes": dag_partition.exchange_bytes,
            "warps": warp_reports,
        }
    result["functions"] = function_reports
    result["exchange_bytes"] = sum(part.exchange_bytes for part in partitions)
    return result


def generate(
    input_path: Path,
    output_dir: Path,
    *,
    warps: int = 8,
    threshold: int = 400,
    max_exchange_bytes: int = 8192,
) -> dict[str, object]:
    if warps <= 0:
        raise DagError("warps must be positive")
    source_bytes = input_path.read_bytes()
    source = source_bytes.decode("utf-8")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    dags = {
        name: parse_function(source, name) for name in MANIFEST
    }
    partitions = {
        name: partition(dag, warps, threshold) for name, dag in dags.items()
    }
    total_exchange = sum(part.exchange_bytes for part in partitions.values())
    if total_exchange > max_exchange_bytes:
        raise DagError(
            f"exchange arena requires {total_exchange} bytes, exceeds "
            f"{max_exchange_bytes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    emit_file(
        output_dir / "slices_jac.mojo",
        source_hash,
        [dags["jac_nuc"]],
        [partitions["jac_nuc"]],
        warps,
        threshold,
    )
    emit_file(
        output_dir / "slices_rhs.mojo",
        source_hash,
        [dags["rhs_specie"], dags["rhs_eint"]],
        [partitions["rhs_specie"], partitions["rhs_eint"]],
        warps,
        threshold,
    )
    report = make_report(
        source_hash,
        list(dags.values()),
        list(partitions.values()),
        warps,
        threshold,
    )
    report["generated_source_bytes"] = {
        name: (output_dir / name).stat().st_size
        for name in ("slices_jac.mojo", "slices_rhs.mojo")
    }
    (output_dir / "slice_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=Path("reproducer.mojo"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=400)
    parser.add_argument("--max-exchange-bytes", type=int, default=8192)
    args = parser.parse_args()
    try:
        report = generate(
            args.input,
            args.output_dir,
            warps=args.warps,
            threshold=args.threshold,
            max_exchange_bytes=args.max_exchange_bytes,
        )
    except DagError as error:
        parser.error(str(error))
    print(json.dumps({"exchange_bytes": report["exchange_bytes"]}))


if __name__ == "__main__":
    main()
