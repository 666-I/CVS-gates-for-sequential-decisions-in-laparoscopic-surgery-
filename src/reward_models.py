"""Reward model variants for 04_reward_model.py and eval_reward_models.py. CPU only.

The reward model predicts the gain of calling AdaptNet on a frame, g_t = score_a - score_r (before
the 0.5 fee), from the state features. Every model here has the same two methods,
fit(frames) and predict(frames), where `frames` is a table with the feature columns and `gain`.

Hand-built risk features (p_cvs = r1 * r2 * r3 is ResNet's own chance that all three criteria hold):
  claim_risk = r_cvs * (1 - p_cvs)    chance that a ResNet CVS claim is false
  miss_risk  = (1 - r_cvs) * p_cvs    chance that ResNet missed a real CVS
Nearly all of the value of a call sits on the frames where ResNet claims CVS: a false claim is fixed
by AdaptNet (gain around +10), a true claim is usually lost (gain around -7). So claim_risk is the
one feature that matters most.

Models:
  gbr200    the original: 200 boosted trees, depth 3
  gbr_small 60 boosted trees, depth 2, bigger leaves (less room to overfit 41 videos)
  ridge     linear model on scaled features
  hgb_mono  histogram boosted trees, depth 2, gain forced to rise with claim_risk
  split     two parts. On claim frames: a logistic regression says how likely the claim is false, and
            gain = p * (mean gain of false claims) + (1 - p) * (mean gain of true claims), with both
            means taken from the fitting frames. On other frames: the mean gain of that position bin.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as c

RISK_COLS = ["p_cvs", "claim_risk", "miss_risk"]
# slim drops the three raw probabilities and frames_left, which the boosted model overfits.
# inter adds claim x position and claim x recent claims, so that a linear model can say "a claim
# late in the video, in a run of claims, is probably true".
FEATURE_SETS = {
    "base": c.FEATURE_COLS,
    "risk": c.FEATURE_COLS + RISK_COLS,
    "compact": ["r_cvs", "p_cvs", "claim_risk", "miss_risk", "r_min", "pos", "recent_claims"],
    "slim": ["r_cvs", "claim_risk", "miss_risk", "pos", "recent_claims"],
    "inter": ["r_cvs", "claim_risk", "miss_risk", "pos", "recent_claims", "claim_pos", "claim_recent"],
}
MODELS = ["gbr200", "gbr_small", "ridge", "hgb_mono", "split"]


def add_risk_features(df):
    """Copy of df with p_cvs, claim_risk and miss_risk added (from r1..r3 and r_cvs only)."""
    df = df.copy()
    df["p_cvs"] = df["r1"] * df["r2"] * df["r3"]
    df["claim_risk"] = df["r_cvs"] * (1 - df["p_cvs"])
    df["miss_risk"] = (1 - df["r_cvs"]) * df["p_cvs"]
    df["claim_pos"] = df["r_cvs"] * df["pos"]
    df["claim_recent"] = df["r_cvs"] * df["recent_claims"]
    return df


class Regressor:
    """A scikit-learn regressor on a fixed list of feature columns."""

    def __init__(self, estimator, features):
        self.estimator, self.features = estimator, list(features)

    def fit(self, frames):
        self.estimator.fit(frames[self.features], frames["gain"])
        return self

    def predict(self, frames):
        return self.estimator.predict(frames[self.features])

    def importances(self):
        est = self.estimator[-1] if hasattr(self.estimator, "steps") else self.estimator
        if hasattr(est, "feature_importances_"):
            return dict(zip(self.features, est.feature_importances_))
        if hasattr(est, "coef_"):
            return dict(zip(self.features, est.coef_))
        return {}


class SplitGain:
    """Claim frames: P(claim is false) times the two mean gains. Other frames: mean gain per position bin.

    The classifier sees the chosen feature set without r_cvs, claim_risk and miss_risk (they are
    constant or duplicates on claim frames). If the fitting frames hold a single kind of claim
    (it happens with a few videos), the classifier is replaced by that base rate.
    """

    def __init__(self, features):
        drop = {"r_cvs", "claim_risk", "miss_risk", "claim_pos", "claim_recent"}
        self.claim_features = [f for f in features if f not in drop]

    @staticmethod
    def _false(frames):
        return (~c.is_cvs(frames[c.Y_COLS].to_numpy())).astype(int)

    def fit(self, frames):
        claims = frames[frames["r_cvs"] == 1]
        false = self._false(claims)
        self.rate = float(false.mean()) if len(false) else 0.5
        self.clf = None
        if 0 < false.sum() < len(false):
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(
                claims[self.claim_features], false)
        is_false = self._false(claims).astype(bool)
        overall = float(claims["gain"].mean()) if len(claims) else 0.0
        self.gain_false = float(claims.loc[is_false, "gain"].mean()) if is_false.any() else overall
        self.gain_true = float(claims.loc[~is_false, "gain"].mean()) if (~is_false).any() else overall
        rest = frames[frames["r_cvs"] == 0]
        self.rest_mean = float(rest["gain"].mean())
        self.bin_mean = rest.groupby("bin")["gain"].mean().to_dict()
        return self

    def predict(self, frames):
        out = frames["bin"].map(self.bin_mean).fillna(self.rest_mean).to_numpy(dtype=float).copy()
        claim = (frames["r_cvs"] == 1).to_numpy()
        if claim.any():
            if self.clf is not None:
                p = self.clf.predict_proba(frames.loc[claim, self.claim_features])[:, 1]
            else:
                p = np.full(claim.sum(), self.rate)
            out[claim] = p * self.gain_false + (1 - p) * self.gain_true
        return out

    def importances(self):
        """Coefficients of the claim classifier (on scaled features, positive = claim more likely false)."""
        return dict(zip(self.claim_features, self.clf[-1].coef_[0])) if self.clf is not None else {}


def make_model(model, features="slim"):
    """Build a model by name. `features` is a key of FEATURE_SETS (split ignores it)."""
    cols = FEATURE_SETS[features]
    if model == "gbr200":
        est = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, min_samples_leaf=20, random_state=0)
    elif model == "gbr_small":
        est = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05,
                                        subsample=0.8, min_samples_leaf=40, random_state=0)
    elif model == "ridge":
        est = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model == "hgb_mono":
        # +1 on claim_risk only. The other directions are not obvious: a high p_cvs can mean a
        # true claim (gain falls) or a miss (gain rises), so they are left free.
        cst = [1 if col == "claim_risk" else 0 for col in cols]
        est = HistGradientBoostingRegressor(max_iter=100, max_depth=2, learning_rate=0.05,
                                            min_samples_leaf=40, l2_regularization=1.0,
                                            monotonic_cst=cst, random_state=0)
    elif model == "split":
        return SplitGain(cols)
    else:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    return Regressor(est, cols)
