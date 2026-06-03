# fxlms-anc

A clean, self-contained **Filtered-x LMS (FxLMS)** active-noise-control implementation in Python, with three runnable experiments that show how it behaves — convergence, spectral effect, and the secondary-path-error failure mode that makes real ANC hard.

No external data required: the experiments **synthesize their own coloured noise**, so every result is reproducible from a clean checkout. You can also point them at a real recording with `--audio`.

> Inspired by [markostam/active-noise-cancellation](https://github.com/markostam/active-noise-cancellation) (a MATLAB/C FxLMS course project). This is an independent Python implementation — the core, path models, noise synthesis, and experiments are original.

## What FxLMS does

Feed-forward ANC: a reference mic hears noise `x`. It reaches your ear through the **primary path** `P` as the disturbance `d = P*x`. A speaker plays anti-noise `y = W·x` that reaches your ear through the **secondary path** `S`. The error mic hears the leftover:

```
e = d − S*y          →  drive e → 0
```

`W` is an adaptive FIR filter. The catch: the control signal passes through `S` before the error mic, which rotates the gradient phase, so plain LMS diverges. The fix is **Filtered-x** — filter the *reference* through an estimate `Ŝ` of the secondary path and use that in the update:

```
x' = Ŝ * x
W += μ · e · x'
```

Everything about how well an ANC system works comes down to how accurate `Ŝ` is.

## Install

```bash
pip install -r requirements.txt
```

## Experiments

```bash
# 1. converge once; render before/after audio + a convergence curve
python experiments/convergence.py --kind brown

# 2. where in the spectrum does it cancel? (it kills the low-frequency drone)
python experiments/spectrum.py --kind brown

# 3. the failure mode: imperfect secondary-path estimate Ŝ
python experiments/secondary_error.py --kind pink

# any of them on a real recording instead of synthetic noise:
python experiments/convergence.py --audio path/to/noise.wav
```

Outputs (wavs + PNGs) land in `outputs/`.

### 1 — Convergence

FxLMS drives the residual down over a second or two. On these signals it reaches **15–28 dB** of broadband reduction depending on how predictable the noise is (steady tonal drone cancels best; broadband transients least).

![convergence](docs/convergence_brownnoise.png)

### 2 — Spectral effect

FxLMS removes the **predictable low-frequency** energy and leaves high-frequency hiss essentially untouched — this is exactly why ANC headphones quiet engine and road noise but not speech.

![spectrum](docs/spectrum_pinknoise.png)

### 3 — Secondary-path error (the hard part)

The update direction comes from the *estimate* `Ŝ`, but the physical world uses the true `S`. Three independent mismatches, with `μ` held fixed:

- **Sign / 180° phase** (`Ŝ = −S`): diverges at any `μ` — the update ascends the gradient. The floor of the ±90° phase-stability rule.
- **Delay / phase error** (`Ŝ = S` shifted): graceful at a safe `μ`, but cliffs hard once the phase error exceeds 90° in the signal's dominant band — and far worse at an aggressive `μ`. Stability margin and `Ŝ`-accuracy margin are the same budget.
- **Gain error** (`Ŝ = c·S`): rescales the effective step to `c·μ`. Underestimate → just slow; overestimate → diverges.

![secondary error](docs/secondary_error_pinknoise.png)

This is what every "adaptive / auto-calibrating" claim in an ANC product is really solving. Real systems either run **online secondary-path identification** (continuously inject a low-level probe and re-learn `Ŝ`) or use **leaky/robust FxLMS** that trades performance for tolerance to `Ŝ` error.

## Layout

```
fxlms_anc/core.py          FxLMS loop, secondary-path ID, path models, noise synthesis
experiments/convergence.py before/after audio + convergence curve
experiments/spectrum.py    PSD before/after + per-band cancellation
experiments/secondary_error.py  Ŝ-mismatch sweeps (sign / delay / gain)
```

## License

MIT — see [LICENSE](LICENSE).
