# Prediction_powered_inference_for_searching_for_rl _paper
# Prediction-Powered Inference on arXiv Papers

1. Project Overview
This project demonstrates Prediction-Powered Inference (PPI), a statistical technique that combines machine learning predictions with a small labeled dataset to producevalid and more precise estimates of a quantity of interest.

Our goal:  
> Estimate the proportion of *Reinforcement Learning (RL)* papers in a sample of 10,000 arXiv abstracts.

---

 Methodology

1. Dataset
- Source: [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) from Hugging Face.
- We used a 10,000 paper subset.
- Each paper has:
  - Title
  - Abstract
  - Metadata (category, authors, etc.)

---

 2. Labeling
- We simulated a 200-paper labeled set
- Papers were labeled `1` if about RL, `0` otherwise.
- Labels were auto-generated for the demo using keyword matching (can be replaced with real manual labels).

---

 3. Model
- Text features: TF-IDF (up to 50,000 features, unigrams & bigrams).
- Classifier: Logistic Regression + isotonic calibration.
- Predictions generated for all 10,000 papers.

---

 4. Estimation Methods Compared
1. **Classical Estimate
   - Uses *only the labeled set* to compute the proportion of RL papers.
   - Confidence Interval (CI) via standard binomial formula.

2. Prediction-Powered Inference (PPI)
   - Uses the mean predicted probability across all papers.
   - Corrects for model bias using the residuals on the labeled set.
   - Produces a valid 95% CI that is often narrower than the classical CI.

---

Results

| Method      | Estimate (%) | 95% CI Lower | 95% CI Upper | CI Width (%) |
|-------------|--------------|--------------|--------------|--------------|
|   PPI       |   1.81       | 0.49         | 3.12         | 2.63         |
| Classical   | 1.50         | 0.00         | 3.18         | 3.18         |

Key Insights:
- PPI CI is narrower → More precision without more labeling.
- Lower bound > 0% for PPI → Avoids the “it could be zero” problem seen in Classical.
- Same labeling cost, better statistical efficiency.

---

Visualizations
Two key plots are generated:

1. Calibration Curve (`calibration_curve.png`)
   - Shows how well the predicted probabilities align with actual outcomes.
   - A well-calibrated model will follow the diagonal.

2. CI Width vs n (`ci_width_vs_n.png`)
   - Demonstrates how CI width decreases with more labeled samples.
   - PPI consistently has **smaller CI width** than Classical.

---

How to Run

1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn torch datasets scikit-learn joblib



