"""Headroom check: how much could routing gain at all? CPU only.

    python src/03_baselines.py

Runs on outputs/prepped.csv, for val and test, at rho = 0.1:
  always ResNet    never call
  always AdaptNet  call on every frame, ignoring the budget (reference only, not a valid policy)
  call-on-claim    call whenever ResNet claims CVS, while budget lasts
  oracle           top B_v frames per video by true gain, where gain > 0.5 (hindsight)

If the oracle is barely above call-on-claim, there is little for a learnt router to win.
Writes outputs/baselines.csv.
"""
import argparse

import numpy as np
import pandas as pd

import common as c

POLICIES = {
    "always ResNet": lambda v: np.zeros(len(v), dtype=int),
    "always AdaptNet (no budget)": lambda v: np.ones(len(v), dtype=int),
    "call-on-claim": lambda v: c.simulate(v, lambda t, b: v["r_cvs"].iat[t] == 1),
    "oracle": c.oracle_calls,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="val", help="val, test or val,test. Test is for the final run only.")
    splits = c.parse_splits(p.parse_args().splits)
    df = pd.read_csv(c.OUTPUTS_DIR / "prepped.csv")
    rows = []
    for split in splits:
        videos = [v.reset_index(drop=True) for _, v in df[df.split == split].groupby("video")]
        for name, policy in POLICIES.items():
            per_video = pd.DataFrame([c.video_results(v, policy(v)) for v in videos])
            rows.append({"split": split, "method": name, **c.reward_settings(), **c.summarise(per_video)})
    out = pd.DataFrame(rows)
    out.to_csv(c.OUTPUTS_DIR / "baselines.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(out.to_string(index=False))
    for split in splits:
        s = out[out.split == split].set_index("method")["mean_return"]
        print(f"{split}: oracle headroom over always ResNet {s['oracle'] - s['always ResNet']:.2f}, "
              f"over call-on-claim {s['oracle'] - s['call-on-claim']:.2f} points per video")


if __name__ == "__main__":
    main()
