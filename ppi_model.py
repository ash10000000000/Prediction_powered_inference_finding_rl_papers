import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from scipy.stats import norm
import joblib

# =====================
# Step 1 — Load Dataset
# =====================
print("Downloading dataset...")
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:10000]")  # 10k sample
df = ds.to_pandas()
df['text'] = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).str.replace('\n', ' ', regex=False)

# ==========================
# Step 2 — Create demo labels
# ==========================
np.random.seed(42)
df = shuffle(df).reset_index(drop=True)

# Randomly mark ~10% as "RL" based on keywords (demo purpose)
keywords = ["reinforcement learning", "RL ", "policy gradient", "q-learning", "actor-critic"]
def auto_label(txt):
    txt_l = txt.lower()
    return int(any(kw in txt_l for kw in keywords))

df['Y'] = df['text'].apply(auto_label)

# Select 200 as "labeled"
labeled_idx = np.random.choice(df.index, size=200, replace=False)
unlabeled_idx = df.index.difference(labeled_idx)

df['is_labeled'] = False
df.loc[labeled_idx, 'is_labeled'] = True

Ldf = df[df['is_labeled']]
Udf = df[~df['is_labeled']]

print(f"Labeled set size: {len(Ldf)}, Unlabeled set size: {len(Udf)}")

# ===============================
# Step 3 — TF-IDF + Logistic Model
# ===============================
print("Training model...")
tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
X_train = tf.fit_transform(Ldf['text'])
y_train = Ldf['Y'].values

base_model = LogisticRegression(max_iter=200, solver='liblinear')
cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
cal_model.fit(X_train, y_train)

joblib.dump(tf, "tfidf.joblib")
joblib.dump(cal_model, "model_calibrated.joblib")

# ======================
# Step 4 — Predictions
# ======================
print("Predicting probabilities...")
X_all = tf.transform(df['text'])
df['Yhat'] = cal_model.predict_proba(X_all)[:, 1]

# =========================================
# Step 5 — PPI vs Classical CI computation
# =========================================
def ppi_mean_ci(Yhat_all, labeled_idx, Y_labeled, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    mu_hat = np.mean(Yhat_all)
    Yhat_L = Yhat_all[labeled_idx]
    residuals = Y_labeled - Yhat_L
    delta_hat = residuals.mean()
    theta = mu_hat + delta_hat
    s_r2 = residuals.var(ddof=1)
    SE = np.sqrt(s_r2 / len(residuals))
    lower = max(0, theta - z * SE)
    upper = min(1, theta + z * SE)
    return lower, upper, theta, SE

alpha = 0.05
ppi_ci = ppi_mean_ci(df['Yhat'].values, labeled_idx, df.loc[labeled_idx, 'Y'].values, alpha)
classical_p = df.loc[labeled_idx, 'Y'].mean()
classical_se = np.sqrt(classical_p * (1 - classical_p) / len(labeled_idx))
z = norm.ppf(1 - alpha / 2)
classical_ci = (max(0, classical_p - z * classical_se), min(1, classical_p + z * classical_se))

print("\n=== Results ===")
print(f"PPI Estimate: {ppi_ci[2]:.4f}, CI: ({ppi_ci[0]:.4f}, {ppi_ci[1]:.4f})")
print(f"Classical Estimate: {classical_p:.4f}, CI: ({classical_ci[0]:.4f}, {classical_ci[1]:.4f})")

# =========================
# Step 6 — Plotting
# =========================
# Calibration curve
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(df.loc[labeled_idx, 'Y'], df.loc[labeled_idx, 'Yhat'], n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Predicted Probability")
plt.ylabel("True Fraction Positive")
plt.title("Calibration Curve (Labeled Set)")
plt.savefig("calibration_curve.png", dpi=150)
plt.close()

# CI width vs n
ns = [50, 100, 150, 200]
widths_ppi = []
widths_classical = []

for n in ns:
    sub_idx = np.random.choice(labeled_idx, size=n, replace=False)
    ppi_ci_tmp = ppi_mean_ci(df['Yhat'].values, sub_idx, df.loc[sub_idx, 'Y'].values, alpha)
    classical_p_tmp = df.loc[sub_idx, 'Y'].mean()
    classical_se_tmp = np.sqrt(classical_p_tmp * (1 - classical_p_tmp) / n)
    classical_ci_tmp = (classical_p_tmp - z * classical_se_tmp, classical_p_tmp + z * classical_se_tmp)
    widths_ppi.append(ppi_ci_tmp[1] - ppi_ci_tmp[0])
    widths_classical.append(classical_ci_tmp[1] - classical_ci_tmp[0])

plt.figure()
sns.lineplot(x=ns, y=widths_ppi, label="PPI CI Width")
sns.lineplot(x=ns, y=widths_classical, label="Classical CI Width")
plt.xlabel("Number of Labeled Samples (n)")
plt.ylabel("CI Width")
plt.title("CI Width vs Labeled Sample Size")
plt.legend()
plt.savefig("ci_width_vs_n.png", dpi=150)
plt.close()

print("\nSaved plots: calibration_curve.png, ci_width_vs_n.png")
