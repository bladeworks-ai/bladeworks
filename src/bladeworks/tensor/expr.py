"""FFmpeg expression parser + torch evaluator (plan task F0).

Architecture map
================

    expression text (registry string, e.g. ``light_deco.BLOOM_EXPRESSION``)
        -> parse()            : libavutil/eval.c grammar -> frozen ``Expr`` tree (cached per text)
        -> evaluate()         : tree walk over an ``Environment`` (scalars, pixel grids, samplers)
                                * scalar-only sub-trees fold in Python doubles (ffmpeg precision)
                                * pixel-dependent sub-trees run as float32 tensor math on device
        -> xfade_custom_rgba(): the ``vf_xfade.c`` custom-transition loop (all four planes at once
                                as a GBRA stack with ``PLANE`` = [4,1,1], nearest ``a0..a3/b0..b3``
                                samplers, uint8 store)
        -> geq_rgba()         : the ``vf_geq.c`` loop (per plane, bilinear ``r/g/b/alpha/p``
                                samplers, ``N``/``T`` frame variables, uint8 store)

Semantics matched against FFmpeg n8.0 sources (``libavutil/eval.c``, ``libavfilter/vf_xfade.c``,
``libavfilter/vf_geq.c``); the goldens in ``experimental_tests/core/test_tensor_expr.py`` and
``test_tensor_transition_golden.py`` run the same strings through the ``ffmpeg`` CLI.

Grammar (eval.c, whitespace stripped first)
-------------------------------------------
    expr    := subexpr (';' subexpr)*                 -- ';' evaluates both, returns the last
    subexpr := term (('+' term) | term_starting_with_'-')*   -- ``a-b`` is ``a + (-b)``
    term    := factor (('*'|'/') factor)*
    factor  := [sign] primary ('^' [sign] primary)*   -- '^' is pow, left-assoc; ``-x^2 = -(x^2)``
    primary := number | name | name '(' expr (',' expr){0,2} ')' | '(' expr ')'
    number  := decimal/exponent/hex per ``av_strtod`` (SI/dB postfixes are rejected loudly)

Every comparison returns 1.0/0.0, ``if(c,a)`` returns 0 when ``c`` is false, ``mod`` is the
*floored* modulo ``d - floor(d/d2)*d2`` (not fmod), ``clip`` returns NaN when ``min > max``,
``between`` is inclusive, division by zero follows C (``x/0 -> +-inf``, ``0/0 -> NaN``),
``round`` is half-away-from-zero, ``*`` doubles as boolean AND.  Unsupported eval.c
features (``st/ld/while/taylor/root/random/randomi/print/time/bitand/bitor/gcd``) raise
``ExpressionError`` at parse time -- no registry expression uses them (grep 2026-08-17).

Value / dtype policy
--------------------
``evaluate`` returns a Python ``float`` for sub-trees that depend on scalars only
(``P, W, H, PLANE, N, T, t, PI`` ...) and a tensor once ``X, Y, A, B`` or a sampler enters.
Scalar folding happens in float64 exactly like ffmpeg (numpy/libm), which removes most float32
drift from the time envelopes (``sin(PI*(1-P))``, ``pow(...)``); pixel math runs in the input
tensors' dtype (float32 on MPS -- float64 is unavailable there).  Measured on the goldens:
float32 pixel math changes ~1e-4 of the stored uint8 values by +-1 (truncation-boundary flips);
float64 on CPU is bit-exact -- its transcendentals (``sin/cos/tan/asin/acos/atan/exp/log/
sinh/cosh/tanh/pow/atan2/hypot``) route through numpy = libm (``_libm_unary`` / ``_libm_binary``)
because torch's vectorized float64 kernels differ from the C library in the last ulp for 5-15%
of arguments, and knife-edge registry expressions are decided there (Clock's ``0.001`` solid
edge, the 360° Circle Wipe ties).  ``if`` with a tensor condition evaluates both branches and
selects with ``torch.where`` (the untaken branch's NaN/inf never propagates); ``if`` with a
scalar condition is lazy (only the taken branch is evaluated), which is what makes the
registry endpoint guards ``if(gte(P,1),A,...)`` free.

uint8 store: C ``dst[x] = (uint8_t)double`` truncates toward zero and is undefined out of
range.  ``quantize_uint8`` truncates, maps NaN to 0 and clamps to [0, 255] -- registry
expressions bound themselves with ``min(255, ...)`` so the clamp is only a safety net; a
golden mismatch (not a silent wrap) is what an unbounded expression produces here.

Main callers:
- ``tensor/transitions.py`` (``xfade_custom`` -> ``xfade_custom_rgba``).
- Effect batches (E1/E2, ``geq_rgba``) once ``geq`` effects are lowered.
- ``experimental_tests/core/test_tensor_expr.py``, ``test_tensor_transition_golden.py``.

Why this exists:
35 of the 52 approved transitions and ~10 effects are one fixed ffmpeg expression over
``A,B,X,Y,W,H,P,PLANE`` (``plan_research/fx_inventory.md`` §0).  One evaluator ports them
mechanically from the registry strings themselves, so a port can never drift from the CPU
reference's expression text; F1/F2 batches append ids, they do not extend this parser.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Sequence, Union

import numpy as np
import torch

Value = Union[float, torch.Tensor]
Sampler = Callable[[Value, Value], torch.Tensor]


class ExpressionError(ValueError):
    """The expression text is outside the supported eval.c grammar (raised at parse time)."""


# --------------------------------------------------------------------------- tree


@dataclass(frozen=True)
class Number:
    value: float


@dataclass(frozen=True)
class Name:
    name: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Node", ...]


@dataclass(frozen=True)
class Binary:
    op: str  # '+', '*', '/', '^', ';'
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Negate:
    operand: "Node"


Node = Union[Number, Name, Call, Binary, Negate]


@dataclass(frozen=True)
class Expr:
    """A parsed expression: the tree plus the names it references."""

    text: str
    root: Node
    names: frozenset[str]
    functions: frozenset[str]


# builtin eval.c constants (looked up after the environment's own names, like ffmpeg)
CONSTANTS: Mapping[str, float] = {"PI": math.pi, "E": math.e, "PHI": (1.0 + 5.0 ** 0.5) / 2.0}

# function -> allowed argument counts
_ARITY: Mapping[str, tuple[int, ...]] = {
    # eval.c one-argument libm functions
    "sinh": (1,), "cosh": (1,), "tanh": (1,), "sin": (1,), "cos": (1,), "tan": (1,),
    "atan": (1,), "asin": (1,), "acos": (1,), "exp": (1,), "log": (1,), "abs": (1,),
    # eval.c typed operators
    "squish": (1,), "gauss": (1,), "isnan": (1,), "isinf": (1,), "floor": (1,), "ceil": (1,),
    "trunc": (1,), "round": (1,), "sgn": (1,), "sqrt": (1,), "not": (1,),
    "mod": (2,), "max": (2,), "min": (2,), "eq": (2,), "gte": (2,), "gt": (2,), "lte": (2,),
    "lt": (2,), "pow": (2,), "hypot": (2,), "atan2": (2,),
    "if": (2, 3), "ifnot": (2, 3), "between": (3,), "clip": (3,), "lerp": (3,),
}
_UNSUPPORTED = frozenset(
    {"st", "ld", "while", "taylor", "root", "random", "randomi", "print", "time",
     "bitand", "bitor", "gcd"}
)
# two-argument sampler functions supplied by the environment (vf_xfade / vf_geq)
SAMPLER_NAMES = frozenset(
    {"a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3", "r", "g", "b", "alpha", "p"}
)

_NUMBER = re.compile(r"(?:0[xX][0-9a-fA-F]+|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _Parser:
    """Recursive descent over eval.c's grammar (one instance per parse call)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.names: set[str] = set()
        self.functions: set[str] = set()

    # -- helpers ------------------------------------------------------------
    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def fail(self, message: str) -> ExpressionError:
        return ExpressionError(f"{message} at offset {self.pos} in {self.text!r}")

    # -- grammar ------------------------------------------------------------
    def parse_expr(self) -> Node:
        node = self.parse_subexpr()
        while self.peek() == ";":
            self.pos += 1
            node = Binary(";", node, self.parse_subexpr())
        return node

    def parse_subexpr(self) -> Node:
        node = self.parse_term()
        while self.peek() in ("+", "-"):
            if self.peek() == "+":
                self.pos += 1
            # '-' stays: the next factor consumes it as its sign (a-b == a+(-b)).
            node = Binary("+", node, self.parse_term())
        return node

    def parse_term(self) -> Node:
        node = self.parse_factor()
        while self.peek() in ("*", "/"):
            op = self.peek()
            self.pos += 1
            node = Binary(op, node, self.parse_factor())
        return node

    def parse_sign(self) -> int:
        char = self.peek()
        if char == "+":
            self.pos += 1
            return 1
        if char == "-":
            self.pos += 1
            return -1
        return 1

    def parse_factor(self) -> Node:
        sign = self.parse_sign()
        node = self.parse_primary()
        while self.peek() == "^":
            self.pos += 1
            sign2 = self.parse_sign()
            exponent = self.parse_primary()
            if sign2 < 0:
                exponent = Negate(exponent)
            node = Binary("^", node, exponent)
        return Negate(node) if sign < 0 else node

    def parse_primary(self) -> Node:
        if self.pos >= len(self.text):
            raise self.fail("unexpected end of expression")
        match = _NUMBER.match(self.text, self.pos)
        if match:
            self.pos = match.end()
            if self.peek() and (self.peek().isalnum() or self.peek() == "_"):
                raise self.fail(f"number {match.group(0)!r} followed by an unsupported postfix")
            literal = match.group(0)
            value = float(int(literal, 16)) if literal[:2].lower() == "0x" else float(literal)
            return Number(value)
        if self.peek() == "(":
            self.pos += 1
            node = self.parse_expr()
            if self.peek() != ")":
                raise self.fail("missing ')'")
            self.pos += 1
            return node
        match = _IDENT.match(self.text, self.pos)
        if not match:
            raise self.fail(f"unexpected character {self.peek()!r}")
        name = match.group(0)
        self.pos = match.end()
        if self.peek() != "(":
            self.names.add(name)
            return Name(name)
        # function call: 1..3 comma-separated expressions
        self.pos += 1
        args = [self.parse_expr()]
        while self.peek() == ",":
            self.pos += 1
            args.append(self.parse_expr())
        if self.peek() != ")":
            raise self.fail(f"missing ')' after arguments of {name}")
        self.pos += 1
        if name in _UNSUPPORTED:
            raise self.fail(f"eval.c function {name!r} is not supported by the tensor evaluator")
        if name in SAMPLER_NAMES:
            if len(args) != 2:
                raise self.fail(f"sampler {name} takes (x, y), got {len(args)} arguments")
        elif name in _ARITY:
            if len(args) not in _ARITY[name]:
                raise self.fail(f"{name} takes {_ARITY[name]} arguments, got {len(args)}")
        else:
            raise self.fail(f"unknown function {name!r}")
        self.functions.add(name)
        return Call(name, tuple(args))


@lru_cache(maxsize=256)
def parse(text: str) -> Expr:
    """Parse ``text`` (cached) into an ``Expr``; raises ``ExpressionError`` on any unsupported syntax."""

    stripped = "".join(ch for ch in text if not ch.isspace())
    if not stripped:
        raise ExpressionError("empty expression")
    parser = _Parser(stripped)
    root = parser.parse_expr()
    if parser.pos != len(stripped):
        raise parser.fail("trailing characters")
    return Expr(text=text, root=root, names=frozenset(parser.names), functions=frozenset(parser.functions))


# --------------------------------------------------------------------------- environment


@dataclass(frozen=True)
class Environment:
    """What an expression can see: named scalars/planes, sampler functions, and the tensor factory.

    ``variables`` map names to floats or tensors (``X``/``Y`` grids, ``A``/``B`` planes);
    ``samplers`` map ``a0..a3/b0..b3/r/g/b/alpha/p`` to ``f(x, y) -> Tensor``.  ``device``
    and ``dtype`` materialize scalars when a tensor is required (``if`` on a tensor condition).
    """

    variables: Mapping[str, Value]
    samplers: Mapping[str, Sampler]
    device: torch.device
    dtype: torch.dtype


def _is_tensor(value: Value) -> bool:
    return isinstance(value, torch.Tensor)


def _tensor(value: Value, env: Environment) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(float(value), device=env.device, dtype=env.dtype)


def _bool(flag: Union[bool, torch.Tensor], env: Environment) -> Value:
    if isinstance(flag, torch.Tensor):
        return flag.to(env.dtype)
    return 1.0 if flag else 0.0


def _scalar_div(a: float, b: float) -> float:
    if b == 0.0:
        return float(a * math.inf) if a == a else math.nan  # C: d * INFINITY (0*inf = NaN)
    return a / b


def _div(a: Value, b: Value) -> Value:
    if not _is_tensor(a) and not _is_tensor(b):
        return _scalar_div(a, b)
    return a / b  # torch: x/0 -> +-inf, 0/0 -> NaN, like C


def _np1(fn: Callable[[np.ndarray], np.ndarray], x: float) -> float:
    with np.errstate(all="ignore"):
        return float(fn(np.float64(x)))


def _np2(fn: Callable[[np.ndarray, np.ndarray], np.ndarray], x: float, y: float) -> float:
    with np.errstate(all="ignore"):
        return float(fn(np.float64(x), np.float64(y)))


def _cmp(op: str, a: Value, b: Value, env: Environment) -> Value:
    if not _is_tensor(a) and not _is_tensor(b):
        return _bool({"lt": a < b, "lte": a <= b, "gt": a > b, "gte": a >= b, "eq": a == b}[op], env)
    ta, tb = _tensor(a, env), _tensor(b, env)
    return _bool({"lt": ta < tb, "lte": ta <= tb, "gt": ta > tb, "gte": ta >= tb, "eq": ta == tb}[op], env)


def _select(cond: Value, then_value: Value, else_value: Value, env: Environment) -> Value:
    if not _is_tensor(cond):
        return then_value if bool(cond) else else_value  # NaN is truthy, as in C
    return torch.where(cond != 0, _tensor(then_value, env), _tensor(else_value, env))


_UNARY_SCALAR: Mapping[str, Callable[[float], float]] = {
    "sinh": lambda x: _np1(np.sinh, x), "cosh": lambda x: _np1(np.cosh, x),
    "tanh": lambda x: _np1(np.tanh, x), "sin": lambda x: _np1(np.sin, x),
    "cos": lambda x: _np1(np.cos, x), "tan": lambda x: _np1(np.tan, x),
    "atan": lambda x: _np1(np.arctan, x), "asin": lambda x: _np1(np.arcsin, x),
    "acos": lambda x: _np1(np.arccos, x), "exp": lambda x: _np1(np.exp, x),
    "log": lambda x: _np1(np.log, x), "abs": lambda x: abs(x),
    "sqrt": lambda x: _np1(np.sqrt, x), "floor": lambda x: _np1(np.floor, x),
    "ceil": lambda x: _np1(np.ceil, x), "trunc": lambda x: _np1(np.trunc, x),
    "round": lambda x: _np1(lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5), x),
    "squish": lambda x: _np1(lambda v: 1.0 / (1.0 + np.exp(4.0 * v)), x),
    "gauss": lambda x: _np1(lambda v: np.exp(-v * v / 2.0) / np.sqrt(2.0 * np.pi), x),
    "sgn": lambda x: float((x > 0) - (x < 0)),
    "not": lambda x: 1.0 if x == 0 else 0.0,
    "isnan": lambda x: 1.0 if x != x else 0.0,
    "isinf": lambda x: 1.0 if x in (math.inf, -math.inf) else 0.0,
}


def _libm_unary(
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    numpy_fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A tensor transcendental that is libm-faithful on CPU float64 (see ``_libm_binary``)."""

    def apply(value: torch.Tensor) -> torch.Tensor:
        if value.dtype == torch.float64 and value.device.type == "cpu":
            with np.errstate(all="ignore"):
                return torch.from_numpy(numpy_fn(value.numpy()))
        return torch_fn(value)

    return apply


def _libm_binary(
    torch_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    numpy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """A two-argument tensor transcendental that is libm-faithful on CPU float64.

    Why this exists: torch's CPU float64 ``sin``/``cos``/``exp``/``atan2``/``pow``/... are
    vectorized (SLEEF-style) and differ from the C library in the last ulp for ~5-15% of
    arguments; numpy's float64 loops call libm and are bit-identical to ``math.*`` (measured
    on macOS arm64).  ffmpeg's ``eval.c`` calls libm, so the CPU float64 path -- the goldens'
    semantics proof and the runtime for float64-only expressions -- routes through numpy.
    Knife-edge registry expressions (Clock's ``0.001`` solid edge, the 360° Circle Wipe's
    great-circle ``lte`` on exact pixel ties) are decided by that last ulp.  Float32 / MPS
    evaluation is untouched.
    """

    def apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dtype == torch.float64 and b.dtype == torch.float64 and a.device.type == "cpu" and b.device.type == "cpu":
            with np.errstate(all="ignore"):
                return torch.from_numpy(numpy_fn(a.numpy(), b.numpy()))
        return torch_fn(a, b)

    return apply


_TENSOR_POW = _libm_binary(torch.pow, np.power)
_TENSOR_ATAN2 = _libm_binary(torch.atan2, np.arctan2)
_TENSOR_HYPOT = _libm_binary(torch.hypot, np.hypot)

_UNARY_TENSOR: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sinh": _libm_unary(torch.sinh, np.sinh), "cosh": _libm_unary(torch.cosh, np.cosh),
    "tanh": _libm_unary(torch.tanh, np.tanh), "sin": _libm_unary(torch.sin, np.sin),
    "cos": _libm_unary(torch.cos, np.cos), "tan": _libm_unary(torch.tan, np.tan),
    "atan": _libm_unary(torch.atan, np.arctan), "asin": _libm_unary(torch.asin, np.arcsin),
    "acos": _libm_unary(torch.acos, np.arccos), "exp": _libm_unary(torch.exp, np.exp),
    "log": _libm_unary(torch.log, np.log), "abs": torch.abs,
    "sqrt": torch.sqrt, "floor": torch.floor, "ceil": torch.ceil, "trunc": torch.trunc,
    "round": lambda v: torch.sign(v) * torch.floor(torch.abs(v) + 0.5),  # C round(), not banker's
    "squish": lambda v: 1.0 / (1.0 + torch.exp(4.0 * v)),
    "gauss": lambda v: torch.exp(-v * v / 2.0) / math.sqrt(2.0 * math.pi),
    "sgn": lambda v: torch.sign(v),
    "not": lambda v: (v == 0).to(v.dtype),
    "isnan": lambda v: torch.isnan(v).to(v.dtype),
    "isinf": lambda v: torch.isinf(v).to(v.dtype),
}


def _pow(a: Value, b: Value, env: Environment) -> Value:
    if not _is_tensor(a) and not _is_tensor(b):
        return _np2(np.power, a, b)
    return _TENSOR_POW(_tensor(a, env), _tensor(b, env))


def _call(name: str, args: Sequence[Value], env: Environment) -> Value:
    if name in _UNARY_SCALAR:
        (x,) = args
        return _UNARY_SCALAR[name](x) if not _is_tensor(x) else _UNARY_TENSOR[name](x)
    if name in ("lt", "lte", "gt", "gte", "eq"):
        return _cmp(name, args[0], args[1], env)
    if name == "pow":
        return _pow(args[0], args[1], env)
    if name in ("max", "min"):
        a, b = args
        # eval.c: d > d2 ? d : d2  (and d < d2 ? d : d2); NaN falls through to d2.
        if not _is_tensor(a) and not _is_tensor(b):
            return a if (a > b if name == "max" else a < b) else b
        ta, tb = _tensor(a, env), _tensor(b, env)
        return torch.where(ta > tb if name == "max" else ta < tb, ta, tb)
    if name == "mod":
        a, b = args
        quotient = _div(a, b)
        floored = _UNARY_SCALAR["floor"](quotient) if not _is_tensor(quotient) else torch.floor(quotient)
        return a - floored * b
    if name == "hypot":
        a, b = args
        if not _is_tensor(a) and not _is_tensor(b):
            return _np2(np.hypot, a, b)
        return _TENSOR_HYPOT(_tensor(a, env), _tensor(b, env))
    if name == "atan2":
        a, b = args
        if not _is_tensor(a) and not _is_tensor(b):
            return _np2(np.arctan2, a, b)
        return _TENSOR_ATAN2(_tensor(a, env), _tensor(b, env))
    if name == "between":
        x, lo, hi = args
        if not any(_is_tensor(v) for v in args):
            return _bool(x >= lo and x <= hi, env)
        tx = _tensor(x, env)
        return _bool((tx >= _tensor(lo, env)) & (tx <= _tensor(hi, env)), env)
    if name == "clip":
        x, lo, hi = args
        if not any(_is_tensor(v) for v in args):
            if x != x or lo != lo or hi != hi or lo > hi:
                return math.nan
            return lo if x < lo else hi if x > hi else x
        tx = _tensor(x, env)
        clipped = torch.minimum(torch.maximum(tx, _tensor(lo, env)), _tensor(hi, env))
        # av_clipd leaves NaN x as NaN (maximum/minimum propagate NaN too); min > max -> NaN.
        if _is_tensor(lo) or _is_tensor(hi):
            bad = _tensor(lo, env) > _tensor(hi, env)
            return torch.where(bad, torch.full_like(clipped, math.nan), clipped)
        if lo != lo or hi != hi or lo > hi:
            return torch.full_like(clipped, math.nan)
        return clipped
    if name == "lerp":
        v0, v1, f = args
        return v0 + (v1 - v0) * f
    raise AssertionError(f"unhandled function {name}")  # parse() validated the name


def _evaluate(node: Node, env: Environment) -> Value:
    if isinstance(node, Number):
        return node.value
    if isinstance(node, Name):
        if node.name in env.variables:
            return env.variables[node.name]
        if node.name in CONSTANTS:
            return CONSTANTS[node.name]
        raise ExpressionError(f"unknown name {node.name!r} (environment has {sorted(env.variables)})")
    if isinstance(node, Negate):
        return -_evaluate(node.operand, env)
    if isinstance(node, Binary):
        left = _evaluate(node.left, env)
        if node.op == ";":
            return _evaluate(node.right, env)
        right = _evaluate(node.right, env)
        if node.op == "+":
            return left + right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return _div(left, right)
        if node.op == "^":
            return _pow(left, right, env)
        raise AssertionError(node.op)
    # Call
    name = node.name
    if name in ("if", "ifnot"):
        cond = _evaluate(node.args[0], env)
        truthy = (name == "if")
        if not _is_tensor(cond):
            taken = bool(cond) == truthy
            if taken:
                return _evaluate(node.args[1], env)
            return _evaluate(node.args[2], env) if len(node.args) == 3 else 0.0
        then_value = _evaluate(node.args[1], env)
        else_value = _evaluate(node.args[2], env) if len(node.args) == 3 else 0.0
        if not truthy:
            then_value, else_value = else_value, then_value
        return _select(cond, then_value, else_value, env)
    if name in SAMPLER_NAMES:
        sampler = env.samplers.get(name)
        if sampler is None:
            raise ExpressionError(f"sampler {name!r} is not available in this environment")
        return sampler(_evaluate(node.args[0], env), _evaluate(node.args[1], env))
    return _call(name, [_evaluate(arg, env) for arg in node.args], env)


def evaluate(expr: Expr, env: Environment) -> Value:
    """Evaluate ``expr`` in ``env``: a float when nothing pixel-dependent is touched, else a tensor."""

    return _evaluate(expr.root, env)


def evaluate_image(expr: Expr, env: Environment, shape: tuple[int, ...]) -> torch.Tensor:
    """``evaluate`` broadcast to a full tensor of ``shape`` (``[H, W]`` or ``[planes, H, W]``) in ``env.dtype``."""

    value = evaluate(expr, env)
    if _is_tensor(value):
        return value.to(env.dtype).expand(shape) if value.shape != torch.Size(shape) else value.to(env.dtype)
    return torch.full(shape, float(value), device=env.device, dtype=env.dtype)


# --------------------------------------------------------------------------- grids + samplers

_GRIDS: dict[tuple[int, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def pixel_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``(X, Y)`` integer pixel-index grids as ``[H, W]`` tensors (cached per size/device/dtype)."""

    key = (height, width, str(device), dtype)
    grids = _GRIDS.get(key)
    if grids is None:
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grids = (grid_x.contiguous(), grid_y.contiguous())
        _GRIDS[key] = grids
    return grids


def _index_pair(x: Value, y: Value, plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampler coordinates as tensors of the plane's dtype/device (scalars become 0-d tensors)."""

    tx = x if _is_tensor(x) else torch.tensor(float(x), device=plane.device, dtype=plane.dtype)
    ty = y if _is_tensor(y) else torch.tensor(float(y), device=plane.device, dtype=plane.dtype)
    return tx.to(plane.dtype), ty.to(plane.dtype)


def nearest_sampler(plane: torch.Tensor) -> Sampler:
    """``vf_xfade.c`` ``getpix``: ``src[(int)av_clipd(x,0,w-1) + (int)av_clipd(y,0,h-1)*linesize]``."""

    height, width = plane.shape
    flat = plane.reshape(-1)

    def sample(x: Value, y: Value) -> torch.Tensor:
        tx, ty = _index_pair(x, y, plane)
        xi = tx.clamp(0, width - 1).trunc().long()
        yi = ty.clamp(0, height - 1).trunc().long()
        xi, yi = torch.broadcast_tensors(xi, yi)
        return flat[yi * width + xi]

    return sample


def bilinear_sampler(plane: torch.Tensor) -> Sampler:
    """``vf_geq.c`` ``getpix`` (default ``interpolation=bilinear``): clamp to ``[0, w-1]``,
    ``xi = (int)x``, ``xn = min(xi+1, w-1)``, weights ``x - xi``."""

    height, width = plane.shape
    flat = plane.reshape(-1)

    def sample(x: Value, y: Value) -> torch.Tensor:
        tx, ty = _index_pair(x, y, plane)
        cx = tx.clamp(0, width - 1)
        cy = ty.clamp(0, height - 1)
        xi = cx.trunc()
        yi = cy.trunc()
        fx = cx - xi
        fy = cy - yi
        xi_l = xi.long()
        yi_l = yi.long()
        xn = torch.clamp(xi_l + 1, max=width - 1)
        yn = torch.clamp(yi_l + 1, max=height - 1)
        xi_l, yi_l, xn, yn, fx, fy = torch.broadcast_tensors(xi_l, yi_l, xn, yn, fx, fy)
        top = (1 - fx) * flat[yi_l * width + xi_l] + fx * flat[yi_l * width + xn]
        bottom = (1 - fx) * flat[yn * width + xi_l] + fx * flat[yn * width + xn]
        return (1 - fy) * top + fy * bottom

    return sample


# --------------------------------------------------------------------------- filter loops

# gbrap plane order: index in the ffmpeg plane loop -> RGBA channel index
GBRA_TO_RGBA: tuple[int, int, int, int] = (2, 0, 1, 3)  # gbra[p] is rgba[GBRA_TO_RGBA[p]]
RGBA_TO_GBRA: tuple[int, int, int, int] = (1, 2, 0, 3)  # rgba[c] is gbra[RGBA_TO_GBRA[c]]


def rgba_to_gbra(planes: torch.Tensor) -> torch.Tensor:
    return planes[list(RGBA_TO_GBRA)]


def gbra_to_rgba(planes: torch.Tensor) -> torch.Tensor:
    return planes[list(GBRA_TO_RGBA)]


def quantize_uint8(values: torch.Tensor) -> torch.Tensor:
    """The C ``(uint8_t)double`` store: truncate toward zero; NaN -> 0; clamp to [0, 255] (see module doc).

    No rounding slack, deliberately: a 2**-11 slack before the float32 truncation was tried
    to rescue exact-integer results (``w*A + (1-w)*A``) and made agreement *worse* (Deco 0.02%
    -> 1.5% of values off by one) because ffmpeg's own doubles land just below integers
    systematically (``(1-w)*A + w*158`` with ``w = 0.30000000000000004`` -> 157.99999999999997
    -> 157); plain truncation of the float32 result is the closest match to that reference.
    """

    return torch.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0).trunc().clamp(0.0, 255.0)


def xfade_progress(frame_index: int, frame_count: int) -> float:
    """``vf_xfade.c`` ``P`` for the k-th of F transition frames with ``offset=0``, ``duration=F/fps``.

    ffmpeg computes ``av_clipf(1.f - (float)(pts - start_pts) / duration_pts, 0, 1)`` in
    *float32* and hands the value to the double evaluator; reproducing the float32 rounding
    keeps knife-edge comparisons (``lt(1-P, 0.40)`` at ``k/F = 0.6``) on ffmpeg's side.  With a
    frame-exact timebase ``pts - start_pts = k`` and ``duration_pts = F``.
    """

    if not 0 <= frame_index < frame_count:
        raise ValueError(f"frame_index {frame_index} outside [0, {frame_count})")
    ratio = np.float32(np.float32(frame_index) / np.float32(frame_count))
    progress = np.float32(np.float32(1.0) - ratio)
    return float(min(np.float32(1.0), max(np.float32(0.0), progress)))


def xfade_custom_rgba(
    expr: Expr,
    a_rgba: torch.Tensor,
    b_rgba: torch.Tensor,
    *,
    progress: float,
) -> torch.Tensor:
    """Run ``expr`` the way ``xfade=transition=custom`` does on two ``gbrap`` frames.

    Inputs and output are RGBA ``[4, H, W]`` code values (0..255, any float dtype); ffmpeg's
    loop runs the expression per plane in **GBRA** order with ``PLANE`` = 0..3, ``A``/``B`` the
    current plane, ``X``/``Y`` pixel indices, ``W``/``H`` the frame size, ``P`` = ``progress``,
    and the nearest-neighbour samplers ``a0..a3``/``b0..b3`` over the (GBRA-indexed) planes.
    Here the four planes are evaluated in one broadcast pass (``A``/``B`` = GBRA stacks,
    ``PLANE`` = ``[4, 1, 1]``) -- same values, one traversal.  Every plane -- alpha included --
    goes through the expression, and the result is stored with the uint8 truncation
    (``quantize_uint8``), exactly like the 8-bit ffmpeg path.

    Main callers:
    - ``tensor/transitions.py::xfade_custom``.
    - ``experimental_tests/core/test_tensor_transition_golden.py``.
    """

    if a_rgba.shape != b_rgba.shape or a_rgba.dim() != 3 or a_rgba.shape[0] != 4:
        raise ValueError(f"expected two [4, H, W] RGBA tensors, got {tuple(a_rgba.shape)} and {tuple(b_rgba.shape)}")
    _, height, width = a_rgba.shape
    device, dtype = a_rgba.device, a_rgba.dtype
    a_gbra = rgba_to_gbra(a_rgba)
    b_gbra = rgba_to_gbra(b_rgba)
    grid_x, grid_y = pixel_grid(height, width, device=device, dtype=dtype)
    samplers: dict[str, Sampler] = {}
    for index in range(4):
        samplers[f"a{index}"] = nearest_sampler(a_gbra[index])
        samplers[f"b{index}"] = nearest_sampler(b_gbra[index])
    # All four planes in one evaluation: ``A``/``B`` are the [4, H, W] GBRA stacks and ``PLANE``
    # a [4, 1, 1] tensor, so plane-independent sub-trees (the X/Y fields of Lens Flare) are
    # computed once and ``if(eq(PLANE,0), ...)`` colour literals select per plane by broadcast.
    plane_index = torch.arange(4, device=device, dtype=dtype).view(4, 1, 1)
    env = Environment(
        variables={
            "X": grid_x, "Y": grid_y, "W": float(width), "H": float(height),
            "A": a_gbra, "B": b_gbra, "PLANE": plane_index, "P": float(progress),
        },
        samplers=samplers,
        device=device,
        dtype=dtype,
    )
    out = quantize_uint8(evaluate_image(expr, env, (4, height, width)))
    return gbra_to_rgba(out)


def geq_rgba(
    expressions: Mapping[str, Expr],
    source_rgba: torch.Tensor,
    *,
    frame_number: int,
    time_seconds: float,
) -> torch.Tensor:
    """Run per-channel ``geq`` expressions (``r``, ``g``, ``b``, ``a`` keys) on an RGBA code frame.

    Mirrors ``vf_geq.c`` on ``gbrap``: samplers ``r/g/b/alpha`` read the named plane with the
    default bilinear ``getpix`` (clamped), ``p`` reads the plane being written; variables
    ``X, Y, W, H, N, T`` (``SW``/``SH`` are 1 for full-resolution planes); uint8 store.  A missing
    channel key raises -- no implicit ``lum`` inheritance, callers name all four.

    Main callers:
    - effect batches (E1/E2) once ``geq`` effects are lowered; ``test_tensor_expr.py`` goldens.
    """

    missing = [key for key in ("r", "g", "b", "a") if key not in expressions]
    if missing:
        raise ValueError(f"geq needs expressions for r, g, b and a; missing {missing}")
    if source_rgba.dim() != 3 or source_rgba.shape[0] != 4:
        raise ValueError(f"expected one [4, H, W] RGBA tensor, got {tuple(source_rgba.shape)}")
    _, height, width = source_rgba.shape
    device, dtype = source_rgba.device, source_rgba.dtype
    grid_x, grid_y = pixel_grid(height, width, device=device, dtype=dtype)
    channel_samplers = {
        "r": bilinear_sampler(source_rgba[0]),
        "g": bilinear_sampler(source_rgba[1]),
        "b": bilinear_sampler(source_rgba[2]),
        "alpha": bilinear_sampler(source_rgba[3]),
    }
    out = torch.empty_like(source_rgba)
    for channel_index, key in enumerate(("r", "g", "b", "a")):
        samplers = dict(channel_samplers)
        samplers["p"] = channel_samplers["alpha" if key == "a" else key]
        env = Environment(
            variables={
                "X": grid_x, "Y": grid_y, "W": float(width), "H": float(height),
                "N": float(frame_number), "T": float(time_seconds), "SW": 1.0, "SH": 1.0,
            },
            samplers=samplers,
            device=device,
            dtype=dtype,
        )
        out[channel_index] = quantize_uint8(evaluate_image(expressions[key], env, (height, width)))
    return out
