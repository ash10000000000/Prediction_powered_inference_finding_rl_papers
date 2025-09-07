import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
import joblib

# Step 1 — Load Dataset
print("Downloading dataset...")
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:10000]")
df = ds.to_pandas()
df['text'] = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).str.replace('\n', ' ', regex=False)

# Step 2 — Create demo labels with adjusted noise
np.random.seed(42)
df = shuffle(df).reset_index(drop=True)
keywords = ["reinforcement learning", "RL ", "policy gradient", "q-learning", "actor-critic"]
def auto_label(txt):
    txt_l = txt.lower()
    return int(any(kw in txt_l for kw in keywords))
df['Y'] = df['text'].apply(auto_label)
df['Y'] = df['Y'].apply(lambda x: np.random.choice([0, 1], p=[0.9, 0.1]) if x == 0 else np.random.choice([0, 1], p=[0.2, 0.8]))  # More realistic noise

# Step 3 — PPI CI Function
def ppi_mean_ci(Yhat_all, labeled_idx, Y_labeled, unlabeled_idx, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    Yhat_L = Yhat_all[labeled_idx]
    Yhat_U = Yhat_all[unlabeled_idx]
    mu_hat = np.mean(Yhat_U)
    residuals = Y_labeled - Yhat_L
    delta_hat = residuals.mean()
    theta = mu_hat + delta_hat
    s_r2 = residuals.var(ddof=1) / len(residuals)
    s_u2 = Yhat_U.var(ddof=1) / len(Yhat_U)
    SE = np.sqrt(s_r2 + s_u2)
    lower = max(0, theta - z * SE)
    upper = min(1, theta + z * SE)
    return lower, upper, theta, SE

# Step 4 — Run Trials
num_trials = 100
ppi_widths = []
classical_widths = []
ppi_cis = []
classical_cis = []
ppi_estimates = []
classical_estimates = []
alpha = 0.05
z = norm.ppf(1 - alpha / 2)

print("Running trials...")
for trial in range(num_trials):
    labeled_idx = np.random.choice(df.index, size=500, replace=False)
    y_labeled = df.loc[labeled_idx, 'Y'].values
    if np.sum(y_labeled) < 2 or np.sum(y_labeled) > len(y_labeled) - 2:  # Lowered threshold
        print(f"Skipping trial {trial}: insufficient positive or negative samples")
        continue
    unlabeled_idx = df.index[~df.index.isin(labeled_idx)]
    
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)  # Reduced features
    X_train = tf.fit_transform(df.loc[labeled_idx, 'text'])
    y_train = y_labeled
    base_model = LogisticRegression(max_iter=200, solver='liblinear')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + trial)
    cal_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=skf)
    cal_model.fit(X_train, y_train)
    
    X_all = tf.transform(df['text'])
    Yhat_all = cal_model.predict_proba(X_all)[:, 1]
    
    ppi_ci = ppi_mean_ci(Yhat_all, labeled_idx, y_labeled, unlabeled_idx, alpha)
    classical_p = y_labeled.mean()
    classical_se = np.sqrt(classical_p * (1 - classical_p) / len(labeled_idx)) if classical_p * (1 - classical_p) > 0 else 0
    classical_ci = (max(0, classical_p - z * classical_se), min(1, classical_p + z * classical_se))
    
    ppi_widths.append(ppi_ci[1] - ppi_ci[0])
    classical_widths.append(classical_ci[1] - classical_ci[0])
    ppi_cis.append(ppi_ci)
    classical_cis.append(classical_ci)
    ppi_estimates.append(ppi_ci[2])
    classical_estimates.append(classical_p)

# Step 5 — Results
true_mean = df['Y'].mean()
print("\n=== Results ===")
print(f"Average PPI CI Width: {np.mean(ppi_widths):.4f}")
print(f"Average Classical CI Width: {np.mean(classical_widths):.4f}")
print(f"Average PPI Estimate: {np.mean(ppi_estimates):.4f}")
print(f"Average Classical Estimate: {np.mean(classical_estimates):.4f}")
print(f"PPI Coverage: {np.mean([ci[0] <= true_mean <= ci[1] for ci in ppi_cis]):.4f}")
print(f"Classical Coverage: {np.mean([ci[0] <= true_mean <= ci[1] for ci in classical_cis]):.4f}")
print(f"Proxy True Mean: {true_mean:.4f}")

# Step 6 — Plotting
plt.figure()
sns.lineplot(x=range(len(ppi_widths)), y=ppi_widths, label="PPI CI Width")
sns.lineplot(x=range(len(classical_widths)), y=classical_widths, label="Classical CI Width")
plt.fill_between(range(len(ppi_widths)), [np.mean(ppi_widths) - np.std(ppi_widths)]*len(ppi_widths), 
                 [np.mean(ppi_widths) + np.std(ppi_widths)]*len(ppi_widths), alpha=0.2, color='blue')
plt.fill_between(range(len(classical_widths)), [np.mean(classical_widths) - np.std(classical_widths)]*len(classical_widths), 
                 [np.mean(classical_widths) + np.std(classical_widths)]*len(classical_widths), alpha=0.2, color='orange')
plt.xlabel("Trial")
plt.ylabel("CI Width")
plt.title("CI Width Across Trials")
plt.legend()
plt.savefig("ci_width_trials.png", dpi=150)
plt.close()

from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(df.loc[labeled_idx, 'Y'], Yhat_all[labeled_idx], n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Predicted Probability")
plt.ylabel("True Fraction Positive")
plt.title("Calibration Curve (Labeled Set, Last Trial)")
plt.savefig("calibration_curve.png", dpi=150)
plt.close()

print("\nSaved plots: ci_width_trials.png, calibration_curve.png")
