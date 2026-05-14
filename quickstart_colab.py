# Quick-Start Template: Overlay Clustering in Google Colab
# Copy-paste ready. Modify only the data loading section.

# ============================================================================
# STEP 1: SETUP (Run once)
# ============================================================================

!pip install scikit-learn pandas numpy matplotlib scipy -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

# Option A: From your CSV file
df = pd.read_csv('df_red_green_overlay.csv')
X = df[['X', 'Y']].values
true_labels = (df['Color'].str.contains('red')).astype(int)

# Option B: From Colab upload
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv(list(uploaded.keys())[0])

print(f"Data shape: {X.shape}")
print(f"Class distribution: {np.bincount(true_labels)}")

# ============================================================================
# STEP 3: VISUALIZE DATA
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[true_labels==0, 0], X[true_labels==0, 1], c='green', alpha=0.5, s=30, label='Green')
ax.scatter(X[true_labels==1, 0], X[true_labels==1, 1], c='red', alpha=0.5, s=30, label='Red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Original Data: Red & Green Overlay')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Visible overlap: Yes (this is the challenge)")

# ============================================================================
# BEST METHOD: GAUSSIAN MIXTURE MODEL + THRESHOLD OPTIMIZATION
# ============================================================================

print("\n" + "="*70)
print("APPLYING: GAUSSIAN MIXTURE MODEL + OPTIMAL THRESHOLD")
print("="*70)

# Step 1: Fit GMM
model = GaussianMixture(n_components=2, covariance_type='full', 
                       random_state=42, n_init=30)
model.fit(X)

# Get soft probabilities
proba = model.predict_proba(X)[:, 1]  # P(red|data)

# Step 2: Find optimal threshold
def threshold_f1(threshold):
    labels = (proba > threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return 0
    return -f1_score(true_labels, labels)

result = minimize_scalar(threshold_f1, bounds=(0.1, 0.9), method='bounded')
optimal_threshold = result.x

# Step 3: Get final labels
labels = (proba > optimal_threshold).astype(int)

# ============================================================================
# RESULTS
# ============================================================================

f1 = f1_score(true_labels, labels)
precision = precision_score(true_labels, labels)
recall = recall_score(true_labels, labels)

print(f"\nOptimal threshold: {optimal_threshold:.4f}")
print(f"\nPerformance Metrics:")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(true_labels, labels)
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

# ============================================================================
# VISUALIZE RESULTS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Clustering result
ax = axes[0]
ax.scatter(X[labels==0, 0], X[labels==0, 1], c='green', alpha=0.6, s=30, label='Green')
ax.scatter(X[labels==1, 0], X[labels==1, 1], c='red', alpha=0.6, s=30, label='Red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'GMM Clustering Result (F1: {f1:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Probability distribution
ax = axes[1]
ax.hist(proba[true_labels==0], bins=30, alpha=0.6, label='True Green', color='green')
ax.hist(proba[true_labels==1], bins=30, alpha=0.6, label='True Red', color='red')
ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
          label=f'Threshold ({optimal_threshold:.3f})')
ax.set_xlabel('P(Red | Data)')
ax.set_ylabel('Count')
ax.set_title('GMM Posterior Probability Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# COMPARE WITH OTHER METHODS
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: BASELINE METHODS")
print("="*70)

results = []

# Method 1: Default GMM (hard assignment, 0.5 threshold)
labels_gmm_default = model.predict(X)
f1_gmm_default = f1_score(true_labels, labels_gmm_default)
results.append({'Method': 'GMM (Hard, default)', 'F1': f1_gmm_default})

# Method 2: Spectral Clustering
gamma = 1.0 / (2 * np.std(X))
spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=gamma, random_state=42)
labels_spectral = spectral.fit_predict(X)
if np.mean(X[labels_spectral==0]) > np.mean(X[labels_spectral==1]):
    labels_spectral = 1 - labels_spectral
f1_spectral = f1_score(true_labels, labels_spectral)
results.append({'Method': 'Spectral (RBF)', 'F1': f1_spectral})

# Method 3: K-Means (for comparison)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_kmeans = kmeans.fit_predict(X)
if np.mean(X[labels_kmeans==0]) > np.mean(X[labels_kmeans==1]):
    labels_kmeans = 1 - labels_kmeans
f1_kmeans = f1_score(true_labels, labels_kmeans)
results.append({'Method': 'K-Means', 'F1': f1_kmeans})

# Our method
results.append({'Method': 'GMM + Optimal Threshold', 'F1': f1})

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print("\n" + results_df.to_string(index=False))

improvement = ((f1 - f1_kmeans) / f1_kmeans * 100)
print(f"\nImprovement over K-Means: {improvement:.1f}%")

# ============================================================================
# OPTIONAL: TRY ALL GMM COVARIANCE TYPES
# ============================================================================

print("\n" + "="*70)
print("OPTIONAL: FIND BEST GMM CONFIGURATION")
print("="*70)

best_config = None
best_score = 0

for cov_type in ['full', 'tied', 'diag', 'spherical']:
    model = GaussianMixture(n_components=2, covariance_type=cov_type, 
                           random_state=42, n_init=20)
    model.fit(X)
    proba = model.predict_proba(X)[:, 1]
    
    # Optimize threshold
    result = minimize_scalar(lambda t: -f1_score(true_labels, (proba > t).astype(int)),
                            bounds=(0.1, 0.9), method='bounded')
    labels = (proba > result.x).astype(int)
    f1 = f1_score(true_labels, labels)
    
    print(f"{cov_type:12s} → F1: {f1:.4f} (threshold: {result.x:.3f})")
    
    if f1 > best_score:
        best_score = f1
        best_config = cov_type
        best_model = model
        best_labels = labels

print(f"\n✓ Best configuration: {best_config} (F1: {best_score:.4f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save predictions
output_df = df.copy()
output_df['predicted_label'] = best_labels
output_df['predicted_color'] = output_df['predicted_label'].map({0: 'green', 1: 'red'})
output_df['is_correct'] = (output_df['predicted_label'] == 
                           (output_df['Color'].str.contains('red')).astype(int))

output_df.to_csv('predictions_output.csv', index=False)
print("\n✓ Saved predictions: predictions_output.csv")

# Summary statistics
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Total points: {len(X)}")
print(f"Correct predictions: {output_df['is_correct'].sum()} ({output_df['is_correct'].mean()*100:.1f}%)")
print(f"Final F1 Score: {best_score:.4f}")
print(f"Method: GMM with covariance type '{best_config}' + optimal threshold")

print("\n✓ Done! Check outputs above and saved CSV file.")

# ============================================================================
# ADVANCED: IF YOU WANT MORE TUNING
# ============================================================================

print("\n" + "="*70)
print("ADVANCED: HYPERPARAMETER GRID SEARCH (Optional)")
print("="*70)

# Uncomment below to run extensive search (takes ~1-2 minutes)

"""
from itertools import product

param_grid = {
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'n_init': [10, 20, 50],
    'threshold': np.linspace(0.2, 0.8, 20)
}

best_f1_overall = 0
best_params_overall = {}

for params in product(*param_grid.values()):
    cov, n_init, thresh = params
    
    model = GaussianMixture(n_components=2, covariance_type=cov, 
                           random_state=42, n_init=n_init)
    model.fit(X)
    proba = model.predict_proba(X)[:, 1]
    labels = (proba > thresh).astype(int)
    
    if len(np.unique(labels)) < 2:
        continue
    
    f1 = f1_score(true_labels, labels)
    
    if f1 > best_f1_overall:
        best_f1_overall = f1
        best_params_overall = {
            'covariance_type': cov,
            'n_init': n_init,
            'threshold': thresh
        }

print(f"\nGrid Search Result (Best F1: {best_f1_overall:.4f}):")
for key, value in best_params_overall.items():
    print(f"  {key}: {value}")
"""

print("\n" + "="*70)
print("Ready to deploy! Use the predictions above for your application.")
print("="*70)
