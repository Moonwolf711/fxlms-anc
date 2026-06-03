"""
online_spm.py -- online secondary-path modeling (Eriksson's method).

The offline experiments learn the secondary-path estimate S_hat once, up front,
and freeze it. Real secondary paths drift (headphone re-seated, temperature,
head movement), and a frozen S_hat then has the wrong phase -> cancellation
collapses (see secondary_error.py).

Eriksson's fix: inject a low-level uncorrelated white PROBE v(n) into the speaker
output and run a SECOND adaptive filter that identifies S_hat from v(n) *while*
the main FxLMS controller keeps cancelling. Because v is independent of the noise,
the secondary-path LMS can pull S out of the error signal continuously.

This script forces the issue: the TRUE secondary path SWITCHES at the halfway
point. We compare:
    offline : S_hat frozen at the original path  -> collapses after the switch
    online  : S_hat tracked by the probe         -> recovers

Signals at the error mic (our sign convention, matching core.run_fxlms):
    u(n) = y(n) + v(n)            speaker = anti-noise + probe
    e(n) = d(n) - S * u(n)        residual (d, anti-noise, and probe all present)
Secondary-path modeling error (drives S_hat -> S):
    f(n) = e(n) + S_hat * v(n)    only the v-correlated part, (S_hat - S)*v, is learnable
    S_hat -= mu_s * f(n) * v_vec

    python experiments/online_spm.py --kind pink
"""
import os

import numpy as np
from scipy.signal import fftconvolve

from _common import OUT, base_parser, get_signal
from fxlms_anc.core import decay_fir, estimate_secondary, stable_step, run_fxlms


def run(x, d, M, fs, S1, S2, Sh0, mu_w, mu_s, probe_std, online, seed, switch):
    """One ANC run with a secondary path that switches S1 -> S2 at sample `switch`."""
    N = len(x)
    rng = np.random.default_rng(seed + 1)
    probe = rng.standard_normal(N) * probe_std

    W = np.zeros(M)
    Sh = Sh0.copy()                 # secondary-path estimate (frozen if offline)
    xb = np.zeros(M)                # reference history
    xpb = np.zeros(M)               # filtered-reference history
    ub = np.zeros(M)                # speaker-output history -> true S
    vb = np.zeros(M)                # probe history -> S_hat modeling
    e = np.zeros(N)
    sh_err = np.zeros(N)            # ||Sh - S_true|| over time, for diagnostics

    for n in range(N):
        S = S1 if n < switch else S2
        xb[1:] = xb[:-1]; xb[0] = x[n]
        y = W @ xb                   # anti-noise
        v = probe[n]                 # injected probe
        u = y + v                    # speaker output
        ub[1:] = ub[:-1]; ub[0] = u
        e[n] = d[n] - S @ ub         # residual at the error mic (true path, audible)
        if not np.isfinite(e[n]) or abs(e[n]) > 1e3:
            return None, None

        vb[1:] = vb[:-1]; vb[0] = v
        f = e[n] + Sh @ vb           # probe-compensated error (== d - S*y once Sh~=S)
        if online:                   # --- adapt S_hat from the probe (Eriksson) ---
            Sh -= mu_s * f * vb      # v-correlated part of f is (Sh - S)*v -> Sh -> S

        xp = Sh @ xb                 # filtered reference x' = S_hat * x (current S_hat)
        xpb[1:] = xpb[:-1]; xpb[0] = xp
        W += mu_w * f * xpb          # enhanced Eriksson: control uses probe-compensated f
        sh_err[n] = np.sqrt(np.mean((Sh - S) ** 2))

    return e, sh_err


def reduction(d, e, lo, hi):
    return 10 * np.log10(np.mean(d[lo:hi] ** 2) / (np.mean(e[lo:hi] ** 2) + 1e-12))


def main():
    args = base_parser(__doc__).parse_args()
    args.seconds = max(args.seconds, 24.0)        # converge, switch at 1/3, then recover
    x, label = get_signal(args)
    M, fs, N = args.M, args.fs, len(x)
    switch = N // 3

    Pz = decay_fir(M, 8, 0.004, fs)               # primary path
    S1 = decay_fir(M, 4, 0.0020, fs)              # original secondary path
    S2 = decay_fir(M, 32, 0.0016, fs)             # AFTER re-seat: big delay shift (~3.5 ms)
    d = fftconvolve(x, Pz)[:N]

    Sh0 = estimate_secondary(S1, M, fs, seed=args.seed)   # offline estimate of S1
    mu_w = stable_step(x, Sh0, M) * 0.7                    # a touch conservative (Sh moves)
    mu_s = 1.5 / M                                         # secondary-path ID step
    probe_std = 0.08                                       # probe level (ID SNR vs noise floor)

    # offline: Sh frozen at Sh0.  online: Sh starts at Sh0 and tracks via the probe.
    e_off, _ = run(x, d, M, fs, S1, S2, Sh0, mu_w, mu_s, probe_std, False, args.seed, switch)
    e_on, sh_err = run(x, d, M, fs, S1, S2, Sh0, mu_w, mu_s, probe_std, True, args.seed, switch)

    # physical ceiling of the NEW path: best reduction S2 allows, given a perfect estimate
    Sh2 = estimate_secondary(S2, M, fs, seed=args.seed)
    ceil2 = run_fxlms(x, Pz, S2, Sh2, M, mu_w)[2]

    q0, q1 = int(0.28 * N), int(0.33 * N)          # converged, just before the switch
    q2, q3 = int(0.92 * N), N                       # well after switch (recovered?)
    print(f"{label}: mu_w={mu_w:.2e} mu_s={mu_s:.2e} probe_std={probe_std}")
    print(f"  new path S2 ceiling (perfect S_hat): {ceil2:+.1f} dB  <- best ANY estimate allows")
    for tag, e in [("offline (frozen S_hat)", e_off), ("online  (probe-tracked)", e_on)]:
        if e is None:
            print(f"  {tag}: DIVERGED"); continue
        pre = reduction(d, e, q0, q1)
        post = reduction(d, e, q2, q3)
        print(f"  {tag}: before switch {pre:+5.1f} dB | after switch {post:+5.1f} dB")

    # audio + plot
    if e_on is not None and e_off is not None:
        import soundfile as sf
        g = 0.9 / (np.max(np.abs(d)) + 1e-12)
        sf.write(os.path.join(OUT, f"online_after_{label}.wav"), (e_on * g).astype(np.float32), fs)
        sf.write(os.path.join(OUT, f"offline_after_{label}.wav"), (e_off * g).astype(np.float32), fs)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            win = 600
            ref = np.mean(d ** 2)
            t = np.arange(N) / fs

            def curve(e):
                return 10 * np.log10(fftconvolve(e ** 2, np.ones(win) / win)[:N] / ref + 1e-12)

            fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax[0].plot(t, curve(e_off), label="offline (frozen S_hat)", color="tab:red", lw=0.9)
            ax[0].plot(t, curve(e_on), label="online (probe-tracked)", color="tab:blue", lw=0.9)
            ax[0].hlines(-ceil2, switch / fs, t[-1], color="tab:green", lw=1.2, ls=":",
                         label=f"S2 ceiling, perfect S_hat ({ceil2:+.1f} dB)")
            ax[0].axvline(switch / fs, color="k", lw=1, ls="--")
            ax[0].text(switch / fs, ax[0].get_ylim()[1], " secondary path switches",
                       va="top", fontsize=9)
            ax[0].axhline(0, color="gray", lw=0.5, ls=":")
            ax[0].set_ylabel("residual power (dB re: disturbance)")
            ax[0].set_title(f"Online secondary-path modeling -- {label} "
                            f"(S switches at t={switch/fs:.0f}s)")
            ax[0].legend(); ax[0].grid(alpha=0.3)
            ax[1].plot(t, sh_err, color="tab:green", lw=0.9)
            ax[1].axvline(switch / fs, color="k", lw=1, ls="--")
            ax[1].set_ylabel("||S_hat - S_true||  (RMS)"); ax[1].set_xlabel("time (s)")
            ax[1].set_title("online S_hat tracking error: jumps at the switch, then re-converges")
            ax[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"online_spm_{label}.png"), dpi=110)
            print(f"  wrote online_after_{label}.wav, offline_after_{label}.wav, online_spm_{label}.png")
        except Exception as ex:
            print(f"  (plot skipped: {ex})")


if __name__ == "__main__":
    main()
