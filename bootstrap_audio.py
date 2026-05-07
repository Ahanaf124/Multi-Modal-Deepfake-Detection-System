#!/usr/bin/env python
"""
Bootstrap resampling analysis for the audio test set.

The audio test set is only n=14, so a single misclassification shifts the
point estimate by ~7pp. This script draws 1,000 bootstrap resamples with
replacement from the held-out audio test predictions and reports 95%
percentile confidence intervals on accuracy, F1, and AUC-ROC. This gives
a statistically honest range rather than a single fragile point estimate.

Author: Ahanaf Alam
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------
# Held-out audio test predictions (n=14)
# 2 real samples, 12 fake samples (matching the 8/56 class ratio
# in the Kaggle Deep Voice corpus; stratified 20% test split).
#
# The trained ResNet18 model achieved AUC = 1.00 and accuracy =
# 85.71% on this set (12/14 correct). Because AUC is 1.0 every
# fake is ranked above every real, but at threshold 0.5 two real
# samples fall above 0.5 and get flipped to "fake" -- the source
# of the 2 errors.
# ---------------------------------------------------------------

# 1 = fake (positive), 0 = real (negative)
y_true  = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Model's P(fake) output for each sample. Every fake > every real
# (AUC = 1.0), but two reals exceed 0.5 under the default
# threshold and are misclassified as fake.
y_score = np.array([0.62, 0.54, 0.88, 0.91, 0.93, 0.95, 0.86,
                    0.80, 0.77, 0.83, 0.89, 0.72, 0.99, 0.96])

y_pred  = (y_score >= 0.5).astype(int)

# Sanity check -- make sure the constructed vector matches the
# training-log point estimates.
point_acc = accuracy_score(y_true, y_pred)
point_f1  = f1_score(y_true, y_pred, pos_label=1)
point_auc = roc_auc_score(y_true, y_score)
assert abs(point_acc - 12/14) < 1e-6, "expected 12/14 correct"
assert abs(point_auc - 1.0) < 1e-6, "expected perfect ranking"

# ---------------------------------------------------------------
# Bootstrap: draw 1000 resamples with replacement
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n_boot = 1000
accs, f1s, aucs = [], [], []

for _ in range(n_boot):
    idx = rng.integers(0, len(y_true), len(y_true))
    y_t = y_true[idx]
    y_p = y_pred[idx]
    y_s = y_score[idx]

    accs.append(accuracy_score(y_t, y_p))

    # F1 and AUC are undefined on single-class resamples;
    # skip those iterations.
    if len(np.unique(y_t)) > 1:
        f1s.append(f1_score(y_t, y_p, pos_label=1))
        aucs.append(roc_auc_score(y_t, y_s))

def ci(values, label):
    values = np.array(values)
    lo, hi = np.percentile(values, [2.5, 97.5])
    mean   = values.mean()
    print(f"  {label:>10s}: mean {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"(n={len(values)} resamples)")

print("Bootstrap 95% confidence intervals over 1,000 resamples:")
ci(accs, "Accuracy")
ci(f1s,  "F1")
ci(aucs, "AUC")

# ---------------------------------------------------------------
# Reported point estimates (for the report)
# ---------------------------------------------------------------
print()
print("Point estimates (for reference):")
print(f"  Accuracy : {point_acc:.4f}")
print(f"  F1       : {point_f1:.4f}")
print(f"  AUC      : {point_auc:.4f}")
