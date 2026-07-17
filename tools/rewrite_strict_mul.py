#!/usr/bin/env python3
"""Rewrite generated chemistry products to explicit portable_mul calls."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import sys
import tokenize

import slice_dag


FUNCTIONS = ("rhs_specie", "rhs_eint", "jac_nuc")
VAR_RE = re.compile(r"^var\s+[A-Za-z_][A-Za-z0-9_]*\s*=")
SKIPPED_TOKENS = {
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


class ExpressionParser:
    def __init__(self, expression: str):
        generated = tokenize.generate_tokens(io.StringIO(expression).readline)
        self.tokens = [token.string for token in generated if token.type not in SKIPPED_TOKENS]
        self.position = 0

    def current(self) -> str:
        if self.position == len(self.tokens):
            return ""
        return self.tokens[self.position]

    def accept(self, value: str) -> bool:
        if self.current() != value:
            return False
        self.position += 1
        return True

    def expect(self, value: str) -> None:
        if not self.accept(value):
            raise SyntaxError(f"expected {value!r}, found {self.current()!r}")

    def parse(self) -> str:
        result = self.parse_conditional()
        if self.current():
            raise SyntaxError(f"unexpected token {self.current()!r}")
        return result

    def parse_conditional(self) -> str:
        result = self.parse_or()
        if self.accept("if"):
            condition = self.parse_or()
            self.expect("else")
            alternative = self.parse_conditional()
            return f"({result} if {condition} else {alternative})"
        return result

    def parse_or(self) -> str:
        result = self.parse_and()
        while self.accept("or"):
            result = f"({result} or {self.parse_and()})"
        return result

    def parse_and(self) -> str:
        result = self.parse_comparison()
        while self.accept("and"):
            result = f"({result} and {self.parse_comparison()})"
        return result

    def parse_comparison(self) -> str:
        result = self.parse_sum()
        while self.current() in ("<", "<=", ">", ">=", "==", "!="):
            operator = self.current()
            self.position += 1
            result = f"({result} {operator} {self.parse_sum()})"
        return result

    def parse_sum(self) -> str:
        result = self.parse_term()
        while self.current() in ("+", "-"):
            operator = self.current()
            self.position += 1
            result = f"({result} {operator} {self.parse_term()})"
        return result

    def parse_term(self) -> str:
        result = self.parse_unary()
        while self.current() in ("*", "/"):
            operator = self.current()
            self.position += 1
            rhs = self.parse_unary()
            if operator == "*":
                result = f"portable_mul({result}, {rhs})"
            else:
                result = f"({result} / {rhs})"
        return result

    def parse_unary(self) -> str:
        if self.current() in ("+", "-", "not"):
            operator = self.current()
            self.position += 1
            separator = " " if operator == "not" else ""
            return f"({operator}{separator}{self.parse_unary()})"
        return self.parse_primary()

    def parse_primary(self) -> str:
        if self.accept("("):
            result = self.parse_conditional()
            self.expect(")")
        else:
            result = self.current()
            if not result:
                raise SyntaxError("expected an expression")
            self.position += 1
        while True:
            if self.accept("."):
                attribute = self.current()
                if not attribute:
                    raise SyntaxError("expected an attribute name")
                self.position += 1
                result += "." + attribute
            elif self.accept("("):
                arguments: list[str] = []
                if not self.accept(")"):
                    while True:
                        arguments.append(self.parse_conditional())
                        if self.accept(")"):
                            break
                        self.expect(",")
                result += "(" + ", ".join(arguments) + ")"
            elif self.accept("["):
                index = self.parse_conditional()
                self.expect("]")
                result += f"[{index}]"
            else:
                return result


def rewrite_expression(expression: str) -> str:
    return ExpressionParser(expression.strip()).parse()


def rewrite_statement(statement: slice_dag.Statement) -> str:
    text = statement.text.strip()
    definition = VAR_RE.match(text)
    if definition is not None:
        prefix = text[: definition.end()]
        return prefix + " " + rewrite_expression(text[definition.end() :])
    if text.startswith("return "):
        return "return " + rewrite_expression(text[len("return ") :])
    return rewrite_expression(text)


def function_bounds(lines: list[str], name: str) -> tuple[int, int]:
    start = None
    for index, line in enumerate(lines):
        match = slice_dag.FUNCTION_RE.match(line)
        if match and match.group(1) == name:
            start = index + 1
            break
    if start is None:
        raise slice_dag.DagError(f"missing function {name}")
    end = start
    while end < len(lines):
        line = lines[end]
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        end += 1
    return start, end


def rewrite_source(source: str) -> str:
    lines = source.splitlines()
    for name in reversed(FUNCTIONS):
        start, end = function_bounds(lines, name)
        body = lines[start:end]
        statements = slice_dag.collect_statements(body, start + 1)
        rewritten = ["    " + rewrite_statement(statement) for statement in statements]
        rewritten.append("")
        lines[start:end] = rewritten
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    original = args.source.read_text()
    rewritten = rewrite_source(original)
    if args.check:
        if rewritten != original:
            raise SystemExit(f"{args.source} requires strict-multiply rewriting")
        return
    args.source.write_text(rewritten)


if __name__ == "__main__":
    main()
