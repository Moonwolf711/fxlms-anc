"""
core.py -- a clean, self-contained Filtered-x LMS (FxLMS) active-noise-control core.

Feed-forward ANC model
----------------------
    x        reference noise (what a noise-sensor mic hears)
    P        primary path   (noise source -> error mic)         d = P * x
    S        secondary path (anti-noise speaker -> error mic)
    W        adaptive FIR controller (the thing we learn)       y = W . x
    e        residual at the error mic                          e = d - S * y   -> 0

The "Filtered-x" trick: the control signal passes through S before reaching the
error mic, which rotates the gradient phase. Plain LMS (W += mu*e*x) then has the
wrong sign and diverges. FxLMS filters the *reference* through an estimate S-hat
of the secondary path and uses that in the update:

    x' = S_hat * x
    W += mu * e * x'

The whole behaviour of an ANC system comes down to how accurate S_hat is -- see
experiments/secondary_error.py.

This module has no dependency on any particular audio file: synth_noise() makes
its own coloured noise, so every experiment is reproducible from scratch. Pass a
real recording with --audio if you have one.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, resample_poly

__all__ = [
    "synth_noise", "load_audio", "decay_fir", "default_paths",
    "estimate_secondary", "run_fxlms",
]


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def synth_noise(kind: str, seconds: float, fs: int, seed: int = 0) -> np.ndarray:
    """Generate normalized coloured noise by spectral shaping of white noise.

    kind: 'white' (flat), 'pink' (-3 dB/oct, 1/sqrt(f)), 'brown' (-6 dB/oct, 1/f,
    a decent stand-in for engine/road drone -- mostly low-frequency energy).
    """
    n = int(seconds * fs)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f[0] = f[1]  # avoid div-by-zero at DC
    shape = {
        "white": np.ones_like(f),
        "pink": 1.0 / np.sqrt(f),
        "brown": 1.0 / f,
    }[kind]
    x = np.fft.irfft(spec * shape, n)
    return x / (np.max(np.abs(x)) + 1e-12)


def load_audio(path: str, fs: int, seconds: float | None = None) -> np.ndarray:
    """Load an audio file, mono-sum, resample to fs, optionally trim, normalize."""
    import soundfile as sf
    sig, fs0 = sf.read(path)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    sig = resample_poly(sig, fs, fs0)
    if seconds is not None:
        sig = sig[: int(seconds * fs)]
    return sig / (np.max(np.abs(sig)) + 1e-12)


# --------------------------------------------------------------------------- #
# acoustic paths
# --------------------------------------------------------------------------- #
def decay_fir(length: int, delay: int, tau: float, fs: int) -> np.ndarray:
    """A plausible short acoustic impulse response: pure delay + exponential decay,
    normalized to unit DC gain so primary and secondary paths are comparable."""
    n = np.arange(length)
    h = np.exp(-n / (tau * fs))
    h[:delay] = 0.0
    return h / (np.sum(np.abs(h)) + 1e-12)


def default_paths(M: int, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """(primary, secondary) FIR paths used across the experiments."""
    Pz = decay_fir(M, delay=8, tau=0.004, fs=fs)   # noise source -> ear
    Sz = decay_fir(M, delay=4, tau=0.002, fs=fs)   # speaker -> ear
    return Pz, Sz


# --------------------------------------------------------------------------- #
# secondary-path identification (offline LMS)
# --------------------------------------------------------------------------- #
def estimate_secondary(Sz: np.ndarray, M: int, fs: int,
                       n_train: int = 20000, seed: int = 0) -> np.ndarray:
    """Learn S_hat by driving white noise through the true Sz and running LMS.
    Mirrors the secondary-path ID stage of classic FxLMS implementations."""
    rng = np.random.default_rng(seed)
    train = rng.standard_normal(n_train)
    target = fftconvolve(train, Sz)[:n_train]
    Szh = np.zeros(M)
    buf = np.zeros(M)
    mu = 0.5 / (M * np.var(train) + 1e-9)
    for n in range(n_train):
        buf[1:] = buf[:-1]; buf[0] = train[n]
        err = target[n] - Szh @ buf
        Szh += mu * err * buf
    return Szh


# --------------------------------------------------------------------------- #
# the FxLMS loop
# --------------------------------------------------------------------------- #
def run_fxlms(x: np.ndarray, Pz: np.ndarray, Sz: np.ndarray, Szh: np.ndarray,
              M: int, mu: float):
    """Run feed-forward FxLMS.

    x   : reference noise
    Pz  : true primary path        (used to form the disturbance d = Pz * x)
    Sz  : true secondary path      (anti-noise actually passes through this)
    Szh : estimated secondary path (used ONLY to filter the reference -> x')
    M   : controller length
    mu  : fixed step size

    Returns (d, e, reduction_dB). reduction_dB is None if the filter diverged.
    """
    N = len(x)
    d = fftconvolve(x, Pz)[:N]          # disturbance at the ear ("before")
    xp = fftconvolve(x, Szh)[:N]        # filtered reference x' = S_hat * x

    W = np.zeros(M)
    xb = np.zeros(M)                    # reference history -> controller output
    xpb = np.zeros(M)                   # filtered-ref history -> weight update
    yb = np.zeros(len(Sz))             # controller-output history -> true Sz
    e = np.zeros(N)

    for n in range(N):
        xb[1:] = xb[:-1]; xb[0] = x[n]
        xpb[1:] = xpb[:-1]; xpb[0] = xp[n]
        y = W @ xb                      # anti-noise (controller output)
        yb[1:] = yb[:-1]; yb[0] = y
        e[n] = d[n] - Sz @ yb           # residual at the error mic (true path)
        if not np.isfinite(e[n]) or abs(e[n]) > 1e3:
            return d, e, None           # diverged
        W += mu * e[n] * xpb            # FxLMS update

    half = N // 2
    red = 10 * np.log10(np.mean(d[half:] ** 2) / (np.mean(e[half:] ** 2) + 1e-12))
    return d, e, red


def stable_step(x: np.ndarray, Szh: np.ndarray, M: int, beta: float = 1.2e-3) -> float:
    """Power-scaled fixed step that lands near the stable optimum (~0.03 on these
    signals). Empirically stable to ~0.05, diverges by ~0.1; beta tunes headroom."""
    N = len(x)
    xp = fftconvolve(x, Szh)[:N]
    return beta / (M * np.var(xp) + 1e-12)
