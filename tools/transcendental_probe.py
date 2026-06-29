#!/usr/bin/env python3
"""Compare C++ libm/std math and Mojo std.math over Float64 samples.

The generated chemistry code uses exp, log, sqrt, cbrt, and abs.  This probe
builds equivalent C++ and Mojo programs over the same sampled Float64 inputs,
then reports ULP and relative differences in their outputs.

This is a full-range sampler, not an exhaustive enumerator of every legal
Float64 value.  Exhaustive coverage would require testing roughly 2^64 input
bit patterns, or all finite patterns minus NaN payloads if NaNs are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FUNCTIONS = ("exp", "log", "sqrt", "cbrt", "abs")
MASK64 = (1 << 64) - 1
SIGN_BIT = 1 << 63
FRAC_MASK = (1 << 52) - 1
EXP_MASK = 0x7FF


@dataclass
class Sample:
    bits: int
    tag: str


@dataclass
class Worst:
    value: float = -1.0
    sample: Sample | None = None
    cpp: float = 0.0
    mojo: float = 0.0
    ulp: int = 0
    rel: float = 0.0


@dataclass
class Summary:
    count: int = 0
    comparable: int = 0
    exact: int = 0
    nan_agree: int = 0
    inf_agree: int = 0
    special_mismatch: int = 0
    max_ulp: Worst = None  # type: ignore[assignment]
    max_rel: Worst = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.max_ulp = Worst()
        self.max_rel = Worst()


def bits_to_float(bits: int) -> float:
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def float_to_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def ordered_float_bits(value: float) -> int:
    bits = float_to_bits(value)
    if bits & SIGN_BIT:
        return (~bits) & MASK64
    return bits | SIGN_BIT


def ulp_distance(a: float, b: float) -> int:
    return abs(ordered_float_bits(a) - ordered_float_bits(b))


def rel_distance(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0.0
    return abs(a - b) / max(abs(a), 1.0e-300)


def category(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return "finite"


def double_literal(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0.0"
    return format(value, ".17g")


def cpp_expr(value: float) -> str:
    lit = double_literal(value)
    if lit == "nan":
        return "std::numeric_limits<double>::quiet_NaN()"
    if lit == "inf":
        return "std::numeric_limits<double>::infinity()"
    if lit == "-inf":
        return "-std::numeric_limits<double>::infinity()"
    return lit


def mojo_expr(value: float) -> str:
    lit = double_literal(value)
    if lit == "nan":
        return "(z / z)"
    if lit == "inf":
        return "(1.0 / z)"
    if lit == "-inf":
        return "(-1.0 / z)"
    return lit


def add_sample(samples: dict[int, Sample], bits: int, tag: str) -> None:
    bits &= MASK64
    if bits not in samples:
        samples[bits] = Sample(bits, tag)


def add_float_neighborhood(
    samples: dict[int, Sample], value: float, radius: int, tag: str
) -> None:
    bits = float_to_bits(value)
    for offset in range(-radius, radius + 1):
        add_sample(samples, bits + offset, tag)


def build_samples(args: argparse.Namespace) -> list[Sample]:
    samples: dict[int, Sample] = {}

    edge_bits = [
        0x0000000000000000,
        0x8000000000000000,
        0x0000000000000001,
        0x8000000000000001,
        0x000FFFFFFFFFFFFF,
        0x800FFFFFFFFFFFFF,
        0x0010000000000000,
        0x8010000000000000,
        0x3FEFFFFFFFFFFFFF,
        0x3FF0000000000000,
        0x3FF0000000000001,
        0x4000000000000000,
        0x4024000000000000,
        0x7FEFFFFFFFFFFFFF,
        0xFFEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000000,
        0xFFF8000000000000,
    ]
    for bits in edge_bits:
        add_sample(samples, bits, "edge")

    focus_values = [
        1.0e-300,
        1.0e-200,
        1.0e-100,
        1.0e-20,
        1.0e-10,
        1.0e-5,
        0.1,
        0.5,
        1.0,
        2.0,
        10.0,
        100.0,
        3032.8492447726062,
        8.0172578014656466,
        math.log(10.0),
        709.782712893384,
        710.0,
        -745.1332191019411,
        -708.3964185322641,
        -1.0,
        -10.0,
        -100.0,
        -3032.8492447726062,
    ]
    for value in focus_values:
        add_float_neighborhood(samples, value, args.nearby_ulps, "focus")

    if args.exponent_stride > 0:
        mantissas = [
            0,
            1,
            0x0008000000000000,
            0x000FFFFFFFFFFFFE,
            0x000FFFFFFFFFFFFF,
        ]
        for exponent in range(0, 2048, args.exponent_stride):
            for sign in (0, SIGN_BIT):
                for mantissa in mantissas:
                    bits = sign | (exponent << 52) | mantissa
                    add_sample(samples, bits, "exponent-sweep")

    rng = random.Random(args.seed)
    for _ in range(args.random_samples):
        add_sample(samples, rng.getrandbits(64), "random")

    ordered = sorted(samples.values(), key=lambda sample: sample.bits)
    if args.max_samples > 0 and len(ordered) > args.max_samples:
        keep = dict((sample.bits, sample) for sample in ordered[: args.max_samples])
        for bits in edge_bits:
            add_sample(keep, bits, "edge")
        ordered = sorted(keep.values(), key=lambda sample: sample.bits)
    return ordered


def generate_cpp(samples: list[Sample]) -> str:
    lines = [
        "#include <cmath>",
        "#include <iomanip>",
        "#include <iostream>",
        "#include <limits>",
        "",
        "static void emit(int index, double x) {",
        "  std::cout << index",
        "            << ' ' << std::setprecision(17) << std::exp(x)",
        "            << ' ' << std::setprecision(17) << std::log(x)",
        "            << ' ' << std::setprecision(17) << std::sqrt(x)",
        "            << ' ' << std::setprecision(17) << std::cbrt(x)",
        "            << ' ' << std::setprecision(17) << std::abs(x)",
        "            << '\\n';",
        "}",
        "",
        "int main() {",
        "  std::cout << std::setprecision(17);",
    ]
    for index, sample in enumerate(samples):
        value = bits_to_float(sample.bits)
        lines.append(f"  emit({index}, {cpp_expr(value)});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_mojo(samples: list[Sample]) -> str:
    lines = [
        "from std.math import abs, cbrt, exp, log, sqrt",
        "",
        "def emit(index: Int, x: Float64):",
        "    print(index, exp(x), log(x), sqrt(x), cbrt(x), abs(x))",
        "",
        "def main():",
        "    var z = 0.0",
    ]
    for index, sample in enumerate(samples):
        value = bits_to_float(sample.bits)
        lines.append(f"    emit({index}, {mojo_expr(value)})")
    return "\n".join(lines) + "\n"


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def fail_process(name: str, proc: subprocess.CompletedProcess[str]) -> None:
    print(f"{name} failed with exit code {proc.returncode}", file=sys.stderr)
    if proc.stdout:
        print("--- stdout ---", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
    if proc.stderr:
        print("--- stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    raise SystemExit(proc.returncode)


def parse_output(text: str, sample_count: int, name: str) -> list[list[float]]:
    rows: list[list[float] | None] = [None] * sample_count
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != len(FUNCTIONS) + 1:
            continue
        try:
            index = int(parts[0])
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        if 0 <= index < sample_count:
            rows[index] = values

    missing = [index for index, row in enumerate(rows) if row is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise RuntimeError(f"{name} output missing {len(missing)} rows: {preview}")
    return [row for row in rows if row is not None]


def compare(
    samples: list[Sample], cpp_rows: list[list[float]], mojo_rows: list[list[float]]
) -> dict[str, Summary]:
    summaries = {name: Summary() for name in FUNCTIONS}
    for sample, cpp_values, mojo_values in zip(samples, cpp_rows, mojo_rows):
        for name, cpp_value, mojo_value in zip(FUNCTIONS, cpp_values, mojo_values):
            summary = summaries[name]
            summary.count += 1
            cpp_cat = category(cpp_value)
            mojo_cat = category(mojo_value)
            if cpp_cat != mojo_cat:
                summary.special_mismatch += 1
                continue
            if cpp_cat == "nan":
                summary.nan_agree += 1
                continue
            if cpp_cat in ("+inf", "-inf"):
                summary.inf_agree += 1
                continue

            summary.comparable += 1
            ulp = ulp_distance(cpp_value, mojo_value)
            rel = rel_distance(cpp_value, mojo_value)
            if ulp == 0:
                summary.exact += 1
            if ulp > summary.max_ulp.value:
                summary.max_ulp = Worst(float(ulp), sample, cpp_value, mojo_value, ulp, rel)
            if rel > summary.max_rel.value:
                summary.max_rel = Worst(rel, sample, cpp_value, mojo_value, ulp, rel)
    return summaries


def sample_payload(sample: Sample) -> dict[str, str | int]:
    value = bits_to_float(sample.bits)
    return {
        "bits": f"0x{sample.bits:016x}",
        "value": double_literal(value),
        "tag": sample.tag,
    }


def summary_to_json(summaries: dict[str, Summary]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, summary in summaries.items():
        def worst_payload(worst: Worst) -> dict[str, object]:
            if worst.sample is None:
                return {}
            return {
                "input": sample_payload(worst.sample),
                "cpp": double_literal(worst.cpp),
                "mojo": double_literal(worst.mojo),
                "ulp": worst.ulp,
                "rel": worst.rel,
            }

        payload[name] = {
            "count": summary.count,
            "comparable": summary.comparable,
            "exact": summary.exact,
            "nan_agree": summary.nan_agree,
            "inf_agree": summary.inf_agree,
            "special_mismatch": summary.special_mismatch,
            "max_ulp": worst_payload(summary.max_ulp),
            "max_rel": worst_payload(summary.max_rel),
        }
    return payload


def print_summary(sample_count: int, summaries: dict[str, Summary]) -> None:
    print("coverage sampled_not_exhaustive")
    print(f"samples {sample_count}")
    print("function comparable exact nan_agree inf_agree special_mismatch max_ulp max_rel")
    for name in FUNCTIONS:
        summary = summaries[name]
        print(
            name,
            summary.comparable,
            summary.exact,
            summary.nan_agree,
            summary.inf_agree,
            summary.special_mismatch,
            summary.max_ulp.ulp,
            f"{summary.max_rel.rel:.6e}",
        )
        for label, worst in (("worst_ulp", summary.max_ulp), ("worst_rel", summary.max_rel)):
            if worst.sample is None:
                continue
            sample = sample_payload(worst.sample)
            print(
                f"  {label}: input_bits={sample['bits']} input={sample['value']}"
                f" tag={sample['tag']} cpp={double_literal(worst.cpp)}"
                f" mojo={double_literal(worst.mojo)} ulp={worst.ulp}"
                f" rel={worst.rel:.6e}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument(
        "--exponent-stride",
        type=int,
        default=1,
        help="sample every Nth IEEE exponent bucket; use 0 to disable",
    )
    parser.add_argument("--nearby-ulps", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="cap total samples after generation; 0 means no cap",
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument(
        "--cxxflags",
        default="-std=c++20 -O3",
        help="flags used when compiling the generated C++ probe",
    )
    parser.add_argument(
        "--ldflags",
        default="-lm",
        help="linker flags used when compiling the generated C++ probe",
    )
    parser.add_argument(
        "--mojo",
        default=os.environ.get("MOJO", "pixi run mojo"),
        help="command used to run Mojo source",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    samples = build_samples(args)
    if not samples:
        raise SystemExit("no samples generated")

    tmpdir = Path(tempfile.mkdtemp(prefix="trans_probe_"))
    try:
        cpp_path = tmpdir / "probe.cpp"
        mojo_path = tmpdir / "probe.mojo"
        exe_path = tmpdir / "probe_cpp"
        cpp_path.write_text(generate_cpp(samples))
        mojo_path.write_text(generate_mojo(samples))

        cxx_cmd = (
            [args.cxx]
            + shlex.split(args.cxxflags)
            + [str(cpp_path), "-o", str(exe_path)]
            + shlex.split(args.ldflags)
        )
        proc = run_command(cxx_cmd, tmpdir)
        if proc.returncode != 0:
            fail_process("C++ compile", proc)

        proc = run_command([str(exe_path)], tmpdir)
        if proc.returncode != 0:
            fail_process("C++ probe", proc)
        cpp_rows = parse_output(proc.stdout, len(samples), "C++")

        mojo_cmd = shlex.split(args.mojo) + [str(mojo_path)]
        proc = run_command(mojo_cmd, workspace)
        if proc.returncode != 0:
            fail_process("Mojo probe", proc)
        mojo_rows = parse_output(proc.stdout, len(samples), "Mojo")

        summaries = compare(samples, cpp_rows, mojo_rows)
        print_summary(len(samples), summaries)
        if args.json is not None:
            args.json.write_text(json.dumps(summary_to_json(summaries), indent=2) + "\n")

        if args.keep_temp:
            print(f"kept temporary files in {tmpdir}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
