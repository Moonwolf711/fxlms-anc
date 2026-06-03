"""
spectrum.py -- WHERE in the spectrum FxLMS removes energy.

Runs FxLMS once, then compares the power spectral density of the disturbance
(before) and the residual (after). Shows that a feed-forward FxLMS kills the
predictable low-frequency drone and leaves the high-frequency hiss largely intact
-- the defining characteristic of real ANC.

    python experiments/spectrum.py --kind brown
    python experiments/spectrum.py --audio path/to/airplane.aiff
"""
import os

import numpy as np
from scipy.signal import welch

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

    half = N // 2                                 # converged region only
    nperseg = 1024
    f, Pd = welch(d[half:], fs=fs, nperseg=nperseg)
    _, Pe = welch(e[half:], fs=fs, nperseg=nperseg)
    band_red = 10 * np.log10(Pd / (Pe + 1e-20) + 1e-20)

    # crossover: highest frequency still getting >=3 dB of cancellation
    sig = f[band_red >= 3.0]
    crossover = sig.max() if sig.size else 0.0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].semilogx(f, 10 * np.log10(Pd + 1e-20), label="before (disturbance)", lw=1.2)
        ax[0].semilogx(f, 10 * np.log10(Pe + 1e-20), label="after (residual)", lw=1.2)
        ax[0].set_ylabel("PSD (dB)"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
        ax[0].set_title(f"FxLMS spectral effect -- {label}  ({red:+.1f} dB broadband)")
        ax[1].semilogx(f, band_red, color="tab:green", lw=1.4)
        ax[1].axhline(0, color="k", lw=0.5, ls="--")
        ax[1].axhline(3, color="gray", lw=0.5, ls=":")
        if crossover:
            ax[1].axvline(crossover, color="tab:red", lw=0.8, ls="--",
                          label=f"3 dB crossover ~ {crossover:.0f} Hz")
            ax[1].legend()
        ax[1].set_ylabel("cancellation (dB)"); ax[1].set_xlabel("frequency (Hz)")
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"spectrum_{label}.png"), dpi=110)
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    lo = band_red[f <= 500].mean()
    hi = band_red[f >= 2000].mean()
    print(f"{label}: broadband {red:+.1f} dB | <500 Hz {lo:+.1f} dB | >2 kHz {hi:+.1f} dB "
          f"| 3 dB crossover ~{crossover:.0f} Hz -> outputs/spectrum_{label}.png")


if __name__ == "__main__":
    main()
