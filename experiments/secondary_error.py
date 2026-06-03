"""
secondary_error.py -- FxLMS's real-world failure mode: an imperfect secondary-path
estimate S_hat. Holds mu fixed (tuned on a perfect S_hat) and sweeps three
mismatches, then renders perfect vs degraded audio.

    1. SIGN  : S_hat = -S        -> 180 deg phase error, diverges at any mu.
    2. DELAY : S_hat = S >> k    -> linear phase error, graceful then cliffs.
    3. GAIN  : S_hat = c * S      -> rescales the effective step (c*mu).

    python experiments/secondary_error.py --kind pink
    python experiments/secondary_error.py --audio path/to/pinknoise.aiff
"""
import os

import numpy as np
import soundfile as sf

from _common import OUT, base_parser, get_signal
from fxlms_anc.core import default_paths, estimate_secondary, run_fxlms, stable_step


def shift(h, k, M):
    out = np.zeros_like(h)
    if k >= 0:
        out[k:] = h[: M - k]
    else:
        out[: M + k] = h[-k:]
    return out


def red_of(x, Pz, Sz, Szh, M, mu):
    return run_fxlms(x, Pz, Sz, Szh, M, mu)[2]


def main():
    args = base_parser(__doc__).parse_args()
    args.seconds = min(args.seconds, 8.0)         # sweeps run many times; keep it brisk
    x, label = get_signal(args)
    M, fs = args.M, args.fs

    Pz, Sz = default_paths(M, fs)
    Szh = estimate_secondary(Sz, M, fs, seed=args.seed)   # ~perfect estimate
    mu = stable_step(x, Szh, M)                            # fixed across all runs
    d, e0, red0 = run_fxlms(x, Pz, Sz, Szh, M, mu)
    print(f"{label}: mu={mu:.3e} (fixed)   perfect S_hat: {red0:+.1f} dB")

    r_flip = red_of(x, Pz, Sz, -Szh, M, mu)
    print(f"  sign flip (S_hat=-S): {'DIVERGED' if r_flip is None else f'{r_flip:+.1f} dB'}")

    KS = [-6, -2, 0, 4, 9, 16, 24, 32, 40]
    print("  delay [safe mu | 3x mu]:")
    delay_rows, delay_hot = [], []
    for k in KS:
        r = red_of(x, Pz, Sz, shift(Szh, k, M), M, mu)
        rh = red_of(x, Pz, Sz, shift(Szh, k, M), M, 3 * mu)
        delay_rows.append((k, r)); delay_hot.append((k, rh))
        s = "DIVERGED" if r is None else f"{r:+5.1f} dB"
        sh = "DIVERGED" if rh is None else f"{rh:+5.1f} dB"
        print(f"    k={k:+3d} ({k/fs*1e3:+.2f} ms) -> {s:>8} | {sh:>8}")

    print("  gain:")
    gain_rows = []
    for c in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        r = red_of(x, Pz, Sz, c * Szh, M, mu)
        gain_rows.append((c, r))
        print(f"    c={c:<4} -> {'DIVERGED' if r is None else f'{r:+5.1f} dB'}")

    # render perfect vs a degraded-but-alive delayed estimate
    g = 0.9 / (np.max(np.abs(d)) + 1e-12)
    sf.write(os.path.join(OUT, f"perfectS_{label}.wav"), (e0 * g).astype(np.float32), fs)
    alive = [(k, r) for k, r in delay_rows if r is not None]
    bad_k = min(alive, key=lambda kr: abs(kr[1] - 3.0))[0] if alive else 16
    _, e_bad, r_bad = run_fxlms(x, Pz, Sz, shift(Szh, bad_k, M), M, mu)
    sf.write(os.path.join(OUT, f"badS_delay{bad_k}_{label}.wav"), (e_bad * g).astype(np.float32), fs)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ks = [k for k, _ in delay_rows]
        ax[0].plot(ks, [r if r is not None else np.nan for _, r in delay_rows], "o-", label="safe mu")
        ax[0].plot(ks, [r if r is not None else np.nan for _, r in delay_hot], "s--",
                   color="tab:red", label="3x mu")
        ax[0].axhline(0, color="k", lw=.5, ls="--")
        ax[0].set_title("Delay mismatch -> phase error"); ax[0].set_xlabel("S_hat delay k (samples)")
        ax[0].set_ylabel("noise reduction (dB)"); ax[0].grid(alpha=.3); ax[0].legend()
        cs = [c for c, _ in gain_rows]
        ax[1].plot(cs, [r if r is not None else np.nan for _, r in gain_rows], "s-")
        ax[1].axhline(0, color="k", lw=.5, ls="--")
        ax[1].set_title("Gain mismatch -> effective step c*mu"); ax[1].set_xlabel("S_hat gain c")
        ax[1].grid(alpha=.3)
        fig.suptitle(f"FxLMS sensitivity to secondary-path error -- {label} (mu fixed)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"secondary_error_{label}.png"), dpi=110)
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    print(f"  wrote perfectS_{label}.wav, badS_delay{bad_k}_{label}.wav ({r_bad:+.1f} dB), "
          f"secondary_error_{label}.png")


if __name__ == "__main__":
    main()
