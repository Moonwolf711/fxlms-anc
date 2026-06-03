"""
convergence.py -- run FxLMS once and render before/after audio + a convergence curve.

    python experiments/convergence.py                       # synthetic brown noise
    python experiments/convergence.py --kind pink
    python experiments/convergence.py --audio path/to/noise.wav
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from _common import OUT, base_parser, get_signal
from fxlms_anc.core import default_paths, estimate_secondary, run_fxlms, stable_step


def main():
    args = base_parser(__doc__).parse_args()
    x, label = get_signal(args)
    M, fs, N = args.M, args.fs, len(x)

    Pz, Sz = default_paths(M, fs)
    Szh = estimate_secondary(Sz, M, fs, seed=args.seed)
    mu = stable_step(x, Szh, M)
    d, e, red = run_fxlms(x, Pz, Sz, Szh, M, mu)

    g = 0.9 / (np.max(np.abs(d)) + 1e-12)        # same gain on both -> audible drop
    sf.write(os.path.join(OUT, f"before_{label}.wav"), (d * g).astype(np.float32), fs)
    sf.write(os.path.join(OUT, f"after_{label}.wav"), (e * g).astype(np.float32), fs)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        win = 400
        inst = fftconvolve(e ** 2, np.ones(win) / win)[:N]
        ref = np.mean(d ** 2)
        t = np.arange(N) / fs
        plt.figure(figsize=(9, 4))
        plt.plot(t, 10 * np.log10(inst / ref + 1e-12), lw=0.8)
        plt.axhline(0, color="k", lw=0.5, ls="--")
        plt.title(f"FxLMS convergence -- {label} (M={M}, fs={fs}, mu={mu:.2e}, {red:+.1f} dB)")
        plt.xlabel("time (s)"); plt.ylabel("residual power (dB re: disturbance)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"convergence_{label}.png"), dpi=110)
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    print(f"{label}: mu={mu:.3e}  reduction={red:+.1f} dB  "
          f"-> outputs/before_{label}.wav, after_{label}.wav, convergence_{label}.png")


if __name__ == "__main__":
    main()
