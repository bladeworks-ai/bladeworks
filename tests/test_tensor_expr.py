"""F0 acceptance: ``tensor/expr.py`` (FFmpeg expression parser + torch evaluator).

Three layers, cheapest first:

1. Parser / scalar semantics with no torch pixels: eval.c grammar corners (``-x^2``,
   ``2^3^2``, ``a-b``, hex / exponent literals, ``;``), arity + unsupported-function
   rejection, and the C-faithful scalar rules (floored ``mod``, ``clip`` NaN, inclusive
   ``between``, ``x/0``, ``round`` half-away, ``max``/``min`` NaN fall-through, lazy ``if``).
2. Tensor semantics without ffmpeg: ``torch.where`` selection discards NaN from the
   untaken branch; ``*`` as boolean AND; grid variables broadcast.
3. **ffmpeg ``geq`` goldens** (the F0 acceptance gate): eleven fixture expression sets --
   together covering every supported function and operator -- run through
   ``ffmpeg -vf format=gbrap,geq=r=..:g=..:b=..:a=..`` on a synthetic RGBA plate and through
   ``expr.geq_rgba`` on the same bytes.  Gate: **float64 evaluation is bit-exact** (proves the
   semantics) except the fractional-coordinate bilinear sampler fixture, where ffmpeg's own
   clang FMA contraction of ``getpix`` puts ~0.02% of flat-region pixels 1 code away from
   the exact rational value (verified against ``fractions.Fraction``; ffmpeg is the one off,
   both ways) -- gate ``<= 1`` there; **float32 evaluation is within 1 code** everywhere
   (truncation-boundary flips at mathematically-exact-integer results, e.g.
   ``127+127*cos(2*PI*Y/H)`` at ``Y = H/4``: float32 ``cos`` = -4e-8 -> 126).  The residual
   histograms are printed with ``-s`` and embedded in assertion messages; no gate is loosened
   beyond these two documented cases.  A fourth check exercises ``N``/``T`` over several frames.

Bounded fixtures: C stores the double result as ``(uint8_t)`` -- out-of-range is undefined
(observed on arm64: wraps, ``-1 -> 255``, ``273 -> 17``) while the tensor path clamps, so every
fixture is designed to stay inside [0, 255] (like every registry expression, which bound
themselves with ``min(255, ...)``); a fixture that leaves the range is a fixture bug, not a gate.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from bladeworks.tensor import expr  # noqa: E402
from bladeworks.tensor.expr import (  # noqa: E402
    Environment,
    ExpressionError,
    evaluate,
    geq_rgba,
    parse,
    pixel_grid,
)

WIDTH, HEIGHT = 320, 180


# --------------------------------------------------------------------------- shared helpers
# (imported by test_tensor_transition_golden.py)


def ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        pytest.skip("needs ffmpeg")
    return binary


def synthetic_plate(seed: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A deterministic RGBA uint8 plate: gradients + an off-centre disc + a rectangle,
    semi-transparent regions (128 / sawtooth alpha), and low-amplitude noise so no plane is flat."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    red = xx * 255.0 / (width - 1)
    green = yy * 255.0 / (height - 1)
    blue = (xx + yy) * 255.0 / (width + height - 2)
    alpha = np.full((height, width), 255.0)
    cx, cy = rng.uniform(0.2, 0.8) * width, rng.uniform(0.2, 0.8) * height
    radius = 0.2 * min(width, height)
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
    red = np.where(disc, 255 - red, red)
    alpha = np.where(disc, 128, alpha)
    x0, x1 = rng.uniform(0.1, 0.5) * width, rng.uniform(0.6, 0.9) * width
    y0, y1 = rng.uniform(0.1, 0.4) * height, rng.uniform(0.6, 0.9) * height
    rect = (xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)
    green = np.where(rect, 255 - green, green)
    alpha = np.where(rect, 64 + (xx % 128), alpha)
    noise = rng.integers(0, 8, (height, width))
    plate = np.stack([red, green, blue, alpha], axis=-1) + noise[..., None]
    return np.clip(plate, 0, 255).astype(np.uint8)


def write_plate(path: Path, plate: np.ndarray) -> Path:
    Image.fromarray(plate, "RGBA").save(path)
    return path


def plate_tensor(plate: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """HWC uint8 RGBA -> [4, H, W] code values."""

    return torch.from_numpy(plate).permute(2, 0, 1).contiguous().to(dtype)


def read_gbrap_frames(raw: Path, width: int, height: int) -> torch.Tensor:
    """Raw planar ``gbrap`` bytes -> ``[N, 4, H, W]`` float32 RGBA code values."""

    data = np.fromfile(raw, dtype=np.uint8)
    frame_bytes = 4 * width * height
    if data.size == 0 or data.size % frame_bytes:
        raise AssertionError(f"raw gbrap output has {data.size} bytes, not a multiple of {frame_bytes}")
    frames = data.reshape(-1, 4, height, width).astype(np.float32)
    return torch.from_numpy(frames)[:, list(expr.GBRA_TO_RGBA)]


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"ffmpeg failed ({result.returncode}): {result.stderr[-2000:]}")


def diff_histogram(got: torch.Tensor, ref: torch.Tensor) -> dict[int, int]:
    """{abs code diff: count} between two code-value tensors."""

    diff = (got.float() - ref.float()).abs()
    return {int(v): int((diff == v).sum()) for v in torch.unique(diff).tolist()}


# --------------------------------------------------------------------------- 1. parser + scalars


def _scalar(text: str, **variables: float) -> float:
    env = Environment(variables=variables, samplers={}, device=torch.device("cpu"), dtype=torch.float64)
    value = evaluate(parse(text), env)
    assert not isinstance(value, torch.Tensor), f"{text!r} folded to a tensor"
    return value


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1+2*3", 7.0),
        ("(1+2)*3", 9.0),
        ("-2^2", -4.0),          # sign applies after the pow chain (parse_factor)
        ("2^3^2", 64.0),         # left-associative
        ("2^-1", 0.5),           # signed exponent
        ("7-2-1", 4.0),          # a-b is a+(-b), left to right
        ("2*-3", -6.0),
        ("1e-3*1000", 1.0),
        (".5+0x10", 16.5),
        ("1E2", 100.0),
        ("PI-3.141592653589793", 0.0),
        ("(1;2;3)", 3.0),        # ';' returns the last
        ("if(0,5)", 0.0),        # two-arg if returns 0 when false
        ("if(1,5)", 5.0),
        ("ifnot(0,5,6)", 5.0),
        ("mod(-1,7)", 6.0),      # floored modulo, not fmod
        ("mod(7.5,2)", 1.5),
        ("between(2,2,3)+between(3,2,3)+between(4,2,3)", 2.0),
        ("clip(9,0,5)", 5.0),
        ("not(0)+not(2)", 1.0),
        ("round(-0.5)", -1.0),   # C round: half away from zero
        ("round(2.5)", 3.0),
        ("trunc(-2.7)+ceil(-2.2)+floor(-2.2)", -7.0),
        ("sgn(-4)+sgn(0)+sgn(9)", 0.0),
        ("lerp(10,20,0.25)", 12.5),
        ("hypot(3,4)+atan2(0,-1)-PI", 5.0),
        ("lt(1,2)*gt(3,2)*gte(2,2)*lte(2,2)*eq(1,1)", 1.0),
        ("pow(2,10)+sqrt(16)+abs(-1)+exp(0)+log(1)", 1030.0),
        ("floor(sin(0)+cos(0))", 1.0),
        ("isnan(0/0)+isinf(1/0)+isinf(-1/0)", 3.0),
        ("squish(0)*2", 1.0),
    ],
)
def test_scalar_semantics(text: str, expected: float) -> None:
    assert _scalar(text) == pytest.approx(expected, abs=1e-12)


def test_scalar_nan_and_division_rules() -> None:
    assert math.isnan(_scalar("clip(3,10,5)"))       # min > max -> NaN
    assert math.isnan(_scalar("0/0"))
    assert _scalar("1/0") == math.inf and _scalar("-1/0") == -math.inf
    assert math.isnan(_scalar("mod(3,0)"))
    assert _scalar("max(0/0,3)") == 3.0              # d > d2 false -> d2
    assert math.isnan(_scalar("max(3,0/0)"))         # 3 > NaN false -> NaN
    assert _scalar("min(0/0,3)") == 3.0
    assert _scalar("if(0/0,7,9)") == 7.0             # NaN is truthy in C


def test_scalar_if_is_lazy_and_variables_fold() -> None:
    # The registry endpoint guard: with a scalar condition only the taken branch runs.
    assert _scalar("if(gte(P,1),A,if(lte(P,0),B,7))", P=1.0, A=3.0, B=4.0) == 3.0
    assert _scalar("if(gte(P,1),A,if(lte(P,0),B,7))", P=0.5, A=3.0, B=4.0) == 7.0
    assert _scalar("sin(PI*(1-P))", P=0.75) == pytest.approx(math.sin(math.pi * 0.25))


@pytest.mark.parametrize(
    "text",
    ["st(0,1)", "ld(0)", "while(1,2)", "random(0)", "print(1)", "bitand(1,2)", "gcd(4,6)",
     "foo(1)", "if(1)", "mod(1)", "clip(1,2)", "between(1,2)", "a0(1)", "1+", "(1+2", "1+2)",
     "3PI", "1kB", "", "1--2"],
)
def test_parse_rejects_unsupported_syntax(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse(text)


def test_parse_collects_names_and_functions() -> None:
    parsed = parse("if(gte(P,1),A,min(255,B+a0(X,Y)*PLANE))")
    assert parsed.names == {"P", "A", "B", "X", "Y", "PLANE"}
    assert parsed.functions == {"if", "gte", "min", "a0"}
    with pytest.raises(ExpressionError, match="unknown name"):
        _scalar("Q+1")


# --------------------------------------------------------------------------- 2. tensor semantics


def _image(text: str, dtype: torch.dtype = torch.float32, **variables: float) -> torch.Tensor:
    grid_x, grid_y = pixel_grid(4, 6, device=torch.device("cpu"), dtype=dtype)
    env = Environment(
        variables={"X": grid_x, "Y": grid_y, "W": 6.0, "H": 4.0, **variables},
        samplers={}, device=torch.device("cpu"), dtype=dtype,
    )
    return expr.evaluate_image(parse(text), env, (4, 6))


def test_tensor_if_discards_untaken_nan_and_boolean_and() -> None:
    out = _image("if(lt(X,1),100,255*sqrt(X-1)/sqrt(W))")
    assert torch.isfinite(out).all()
    assert (out[:, 0] == 100).all()
    assert out[0, 5] == pytest.approx(255 * math.sqrt(4) / math.sqrt(6), rel=1e-6)
    band = _image("255*lt(X,W/2)*gt(Y,H/2)")
    assert band[3, 0] == 255 and band[3, 5] == 0 and band[0, 0] == 0
    assert _image("if(gt(X,2),200)")[0].tolist() == [0, 0, 0, 200, 200, 200]  # broadcast scalar branch


def test_tensor_clip_between_mod_match_scalar_rules() -> None:
    assert torch.isnan(_image("clip(X,10,5)")).all()
    assert _image("between(X,1,3)")[0].tolist() == [0, 1, 1, 1, 0, 0]
    assert _image("mod(X-3,4)")[0].tolist() == [1, 2, 3, 0, 1, 2]
    assert _image("round(X-2.5)")[0].tolist() == [-3, -2, -1, 1, 2, 3]  # half away from zero
    assert _image("max(X,2)+min(X,2)")[0].tolist() == [2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize(
    "text,reference",
    [
        ("sin(X*12.9898+Y*78.233)", lambda x, y: math.sin(x * 12.9898 + y * 78.233)),
        ("cos(X*0.7-Y)", lambda x, y: math.cos(x * 0.7 - y)),
        ("exp(-(X+Y)/3)", lambda x, y: math.exp(-(x + y) / 3)),
        ("atan2(Y-1.5,X-2.5)", lambda x, y: math.atan2(y - 1.5, x - 2.5)),
        ("pow(X+0.5,Y/2)", lambda x, y: math.pow(x + 0.5, y / 2)),
        ("acos(clip((X-2.5)/3,-1,1))", lambda x, y: math.acos(min(1.0, max(-1.0, (x - 2.5) / 3)))),
        ("hypot(X-2,Y-1)", lambda x, y: np.hypot(x - 2.0, y - 1.0)),
        ("log(X+1)+tan(Y/5)+atan(X)+sinh(Y/4)", lambda x, y: math.log(x + 1) + math.tan(y / 5) + math.atan(x) + math.sinh(y / 4)),
    ],
)
def test_float64_cpu_transcendentals_are_libm_faithful(text: str, reference) -> None:
    """CPU float64 tensor evaluation must reproduce libm bit for bit (``expr._libm_unary`` /
    ``_libm_binary``): torch's vectorized float64 ``sin/cos/exp/atan2/pow/...`` differ from the C
    library in the last ulp for 5-15% of arguments, which decides knife-edge registry expressions
    (Clock's ``0.001`` solid edge, the 360° Circle Wipe's great-circle ``lte`` on pixel ties)."""

    got = _image(text, torch.float64)
    expected = torch.tensor([[float(reference(float(x), float(y))) for x in range(6)] for y in range(4)], dtype=torch.float64)
    assert torch.equal(got, expected), f"{text}: max |diff| {(got - expected).abs().max().item():.3e} (not libm-exact)"


def test_quantize_uint8_truncates_and_bounds() -> None:
    values = torch.tensor([254.9999, 255.0, 300.0, -0.5, -3.0, math.nan, math.inf, -math.inf])
    assert expr.quantize_uint8(values).tolist() == [254.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0]


def test_xfade_progress_matches_float32_ffmpeg_arithmetic() -> None:
    assert expr.xfade_progress(0, 10) == 1.0
    assert expr.xfade_progress(6, 10) == float(np.float32(1.0) - np.float32(0.6))
    assert expr.xfade_progress(6, 10) != 0.4  # the double value would land on the other side of lt(1-P,0.4)
    with pytest.raises(ValueError):
        expr.xfade_progress(10, 10)


# --------------------------------------------------------------------------- 3. ffmpeg geq goldens

# name -> per-channel expressions.  Together they cover every supported function and operator:
# + - * / ^ ; unary minus, hex/exponent literals, PI, if/ifnot (2- and 3-arg), lt/lte/gt/gte/eq,
# between, not, clip, min, max, mod, floor, ceil, trunc, round, sgn, pow, sqrt, abs, exp, sin,
# cos, atan2, hypot, lerp, squish, gauss, isnan, isinf, and the samplers r/g/b/alpha/p.
GEQ_FIXTURES: Mapping[str, Mapping[str, str]] = {
    "arith_precedence": dict(
        r="255-abs(-(X-W/2))^1.2/3", g="(X+Y)/4+2^3^2/2-1",
        b="(-X^0.5*8+Y*-1)/2+255", a="alpha(X,Y)"),
    "compare_boolean": dict(
        r="255*lt(X,W/2)*gt(Y,H/2)", g="128*gte(X,Y)+64*lte(X,Y)+63*eq(X,Y)",
        b="255*between(X,W/4,3*W/4)*not(between(Y,H/4,H/2))", a="200*not(eq(X,0))+55*gt(alpha(X,Y),128)"),
    "if_forms": dict(
        r="if(lt(X,W/3),r(X,Y),if(lt(X,2*W/3),g(X,Y),b(X,Y)))", g="if(gt(X,W/2),200)",
        b="ifnot(gt(Y,H/2),100,ifnot(gt(X,W/2),50))", a="if(lt(X,1),100,255*sqrt(X-1)/sqrt(W))"),
    "math_funcs": dict(
        r="clip(255*sin(PI*X/W),0,255)", g="127+127*cos(2*PI*Y/H)",
        b="255*exp(-hypot(X-W/2,Y-H/2)/50)", a="clip(128+80*atan2(Y-H/2,X-W/2),0,255)"),
    "mod_floor_pow_sqrt": dict(
        r="mod(X-W/2,7)*30", g="floor(X/13)*10", b="pow(X/W,2.2)*255",
        a="min(255,sqrt(X*Y)+max(0,mod(-Y,5)*10))"),
    "samplers_bilinear": dict(
        r="r(X*0.5+0.25,Y*0.7)", g="g(X-10,Y+1000)", b="b(W-1-X+0.5,Y)",
        a="alpha(X/2+W/4,H-1-Y*0.5)"),
    "sampler_plane_swap": dict(r="g(X,Y)", g="b(X,Y)", b="r(X,Y)", a="p(X,Y)"),
    "p_current_plane": dict(r="p(X,Y)", g="p(X,Y)/2", b="255-p(X,Y)", a="p(X,Y)"),
    "rounding": dict(
        r="round((X-W/2)/4)*3+128", g="trunc(X/4.5)*3", b="ceil(X/4.5)*3",
        a="clip(128+sgn(X-W/2)*100+lerp(0,20,X/W),0,255)"),
    "hex_exponent_last": dict(
        r="0x80+X*1e-1", g="1E2+(X;Y)/2", b="squish((X-W/2)/W)*255", a="gauss((X-W/2)/W*4)*600"),
    "clip_nan_between": dict(
        r="if(isnan(clip(X,10,5)),100,0)", g="255*isinf(1/0)+0*isinf(X)",
        b="min(255,max(0,X/0*0))", a="mod(X,0)*0+255*isnan(mod(X,0))"),
}
# fixtures where the reference itself is 1 code off the exact value (see module doc)
_FLOAT64_TOLERANCE_ONE = frozenset({"samplers_bilinear"})


@pytest.fixture(scope="module")
def plate_png(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    plate = synthetic_plate(1)
    return write_plate(tmp_path_factory.mktemp("geq") / "plate.png", plate), plate


def _geq_filter(channels: Mapping[str, str]) -> str:
    return "format=gbrap,geq=r='{r}':g='{g}':b='{b}':a='{a}'".format(**channels)


@pytest.mark.parametrize("name", sorted(GEQ_FIXTURES))
def test_geq_golden(name: str, plate_png: tuple[Path, np.ndarray], tmp_path: Path) -> None:
    png, plate = plate_png
    channels = GEQ_FIXTURES[name]
    raw = tmp_path / f"{name}.raw"
    run_ffmpeg(["-i", str(png), "-vf", _geq_filter(channels), "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "gbrap", str(raw)])
    reference = read_gbrap_frames(raw, WIDTH, HEIGHT)[0]
    expressions = {key: parse(text) for key, text in channels.items()}
    for dtype, limit in ((torch.float64, 1 if name in _FLOAT64_TOLERANCE_ONE else 0), (torch.float32, 1)):
        got = geq_rgba(expressions, plate_tensor(plate, dtype), frame_number=0, time_seconds=0.0)
        histogram = diff_histogram(got, reference)
        print(f"geq {name} {dtype}: {histogram}")
        assert max(histogram) <= limit, f"{name} {dtype}: |diff| histogram {histogram} exceeds {limit}"


def test_geq_frame_variables_n_and_t(plate_png: tuple[Path, np.ndarray], tmp_path: Path) -> None:
    """``N`` is the output frame count so far, ``T`` its pts in seconds (25 fps: T = N/25)."""

    png, plate = plate_png
    channels = dict(r="clip(N*40,0,255)", g="clip(T*1000,0,255)", b="r(X,Y)*gt(N,2)", a="255")
    raw = tmp_path / "nt.raw"
    run_ffmpeg(["-loop", "1", "-framerate", "25", "-t", "0.2", "-i", str(png), "-vf",
                _geq_filter(channels), "-frames:v", "5", "-f", "rawvideo", "-pix_fmt", "gbrap", str(raw)])
    frames = read_gbrap_frames(raw, WIDTH, HEIGHT)
    assert frames.shape[0] == 5
    expressions = {key: parse(text) for key, text in channels.items()}
    for index in range(5):
        got = geq_rgba(expressions, plate_tensor(plate, torch.float64), frame_number=index, time_seconds=index / 25)
        assert diff_histogram(got, frames[index]) == {0: 4 * WIDTH * HEIGHT}, f"frame {index}"
