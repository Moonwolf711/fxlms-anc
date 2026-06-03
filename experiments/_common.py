"""Shared CLI plumbing for the experiments."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fxlms_anc.core import synth_noise, load_audio  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--kind", default="brown", choices=["white", "pink", "brown"],
                   help="synthetic noise colour (default: brown ~ engine/road drone)")
    p.add_argument("--audio", default=None, help="path to a real recording (overrides --kind)")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--fs", type=int, default=8000)
    p.add_argument("--M", type=int, default=128, help="controller / filter length")
    p.add_argument("--seed", type=int, default=0)
    return p


def get_signal(args):
    """Return (x, label). Real audio if --audio is given, else synthetic noise."""
    if args.audio:
        label = os.path.splitext(os.path.basename(args.audio))[0]
        return load_audio(args.audio, args.fs, args.seconds), label
    return synth_noise(args.kind, args.seconds, args.fs, args.seed), f"{args.kind}noise"
