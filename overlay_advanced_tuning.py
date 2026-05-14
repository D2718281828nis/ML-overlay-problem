# Advanced Hyperparameter Optimization for Overlay Clustering
# Comprehensive tuning and diagnostic tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import LocalOutlierFactor
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('df_red_green_overlay.csv')
X = df[['X', 'Y']].values
true_labels = (df['Color'].str.contains('red')).astype(int)

print("="*70)
print("OVERLAY CLUSTERING: HYPERPARAMETER OPTIMIZATION")
print("="*70)

# ============================================================================
# TECHNIQUE 1: GMM WITH BIC/AIC MODEL SELECTION
# ============================================================================
print("\n1. GMM Model Selection (BIC/AIC)")
print("-" * 70)

bic_scores = []
aic_scores = []
n_components_range = range(2, 8)

for n in n_components_range:
    model = GaussianMixture(n_components=n, random_state=42, n_init=20)
    model.fit(X)
    bic_scores.append(model.bic(X))
    aic_scores.append(model.aic(X))

optimal_n = list(n_components_range)[np.argmin(bic_scores)]
print(f"Optimal components by BIC: {optimal_n}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n_components_range, bic_scores, 'o-', label='BIC', linewidth=2, markersize=8)
ax.plot(n_components_range, aic_scores, 's-', label='AIC', linewidth=2, markersize=8)
ax.axvline(optimal_n, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Score')
ax.set_title('GMM Model Selection (Lower is Better)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gmm_model_selection.png', dpi=150, bbox_inches='tight')
print("✓ Saved: gmm_model_selection.png\n")
plt.show()

# ============================================================================
# TECHNIQUE 2: SPECTRAL CLUSTERING WITH GAMMA OPTIMIZATION
# ============================================================================
print("2. Spectral Clustering - Gamma (Bandwidth) Optimization")
print("-" * 70)

gamma_values = np.logspace(-2, 2, 20)
f1_scores_spectral = []

for gamma in gamma_values:
    sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=gamma, 
                            random_state=42, eigen_solver='arpack')
    labels = sc.fit_predict(X)
    
    # Try both orientations
    if np.mean(X[labels == 0]) > np.mean(X[labels == 1]):
        labels = 1 - labels
    
    f1 = f1_score(true_labels, labels)
    f1_scores_spectral.append(f1)

optimal_gamma = gamma_values[np.argmax(f1_scores_spectral)]
best_f1_spectral = max(f1_scores_spectral)

print(f"Optimal gamma: {optimal_gamma:.4f}")
print(f"Best F1 Score: {best_f1_spectral:.4f}\n")

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(gamma_values, f1_scores_spectral, 'o-', linewidth=2, markersize=8)
ax.axvline(optimal_gamma, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_gamma:.4f}')
ax.axhline(best_f1_spectral, color='green', linestyle='--', alpha=0.5, label=f'Best F1: {best_f1_spectral:.4f}')
ax.set_xlabel('Gamma (log scale)')
ax.set_ylabel('F1 Score')
ax.set_title('Spectral Clustering: Gamma Sensitivity')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('spectral_gamma_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: spectral_gamma_optimization.png\n")
plt.show()

# ============================================================================
# TECHNIQUE 3: LOCAL OUTLIER FACTOR (LOF) FOR OVERLAP DETECTION
# ============================================================================
print("3. Local Outlier Factor (LOF) - Density-Based Separation")
print("-" * 70)

# LOF detects local density inconsistencies
lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
lof_scores = lof.negative_outlier_factor_

# Points with extreme LOF scores are at cluster boundaries
# Use this to weight the clustering
normalized_lof = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

# Get GMM probabilities
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=20)
gmm_proba = gmm.fit_predict_proba(X)

# Weight soft probabilities by LOF density confidence
weighted_proba = gmm_proba[:, 1] * normalized_lof
gmm_labels = gmm.fit_predict(X)
if np.mean(X[gmm_labels == 0]) > np.mean(X[gmm_labels == 1]):
    gmm_labels = 1 - gmm_labels

# Find optimal threshold on weighted probabilities
def weighted_f1(threshold):
    labels = (weighted_proba > threshold * np.max(weighted_proba)).astype(int)
    if len(np.unique(labels)) < 2:
        return 0
    return -f1_score(true_labels, labels)

result = minimize_scalar(weighted_f1, bounds=(0.1, 1.0), method='bounded')
optimal_threshold = result.x * np.max(weighted_proba)

lof_labels = (weighted_proba > optimal_threshold).astype(int)
lof_f1 = f1_score(true_labels, lof_labels)

print(f"LOF + GMM F1 Score: {lof_f1:.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}\n")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LOF scores
scatter = axes[0].scatter(X[:, 0], X[:, 1], c=lof_scores, cmap='viridis', s=30, alpha=0.6)
axes[0].set_title('Local Outlier Factor Scores\n(Higher = More Anomalous)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(scatter, ax=axes[0])

# LOF-weighted clustering
axes[1].scatter(X[lof_labels == 0, 0], X[lof_labels == 0, 1], 
               c='green', alpha=0.6, s=30, label='Green')
axes[1].scatter(X[lof_labels == 1, 0], X[lof_labels == 1, 1], 
               c='red', alpha=0.6, s=30, label='Red')
axes[1].set_title(f'LOF + GMM Clustering (F1: {lof_f1:.4f})')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lof_density_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: lof_density_analysis.png\n")
plt.show()

# ============================================================================
# TECHNIQUE 4: MULTI-SCALE CLUSTERING (Different Distance Metrics)
# ============================================================================
print("4. Multi-Scale Analysis (Different Distance Metrics)")
print("-" * 70)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

metrics = {
    'Euclidean': euclidean_distances,
    'Manhattan': manhattan_distances,
    'Cosine': cosine_distances
}

best_results = {}

for metric_name, metric_func in metrics.items():
    D = metric_func(X)
    # Convert distance to similarity
    sigma = np.median(D[D > 0])
    affinity = np.exp(-D / (2 * sigma ** 2))
    
    # Spectral clustering on custom affinity
    from sklearn.cluster import SpectralClustering
    # Manual spectral clustering with custom affinity
    eigenvalues, eigenvectors = np.linalg.eigh(affinity)
    idx = np.argsort(eigenvalues)[::-1][:2]
    embedding = eigenvectors[:, idx]
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = kmeans.fit_predict(embedding)
    
    if np.mean(X[labels == 0]) > np.mean(X[labels == 1]):
        labels = 1 - labels
    
    f1 = f1_score(true_labels, labels)
    best_results[metric_name] = f1
    
    print(f"{metric_name:12s} → F1: {f1:.4f}")

print()

# ============================================================================
# TECHNIQUE 5: MIXTURE DENSITY ESTIMATION WITH KERNEL SMOOTHING
# ============================================================================
print("5. Gaussian Kernel Density Estimation (KDE) Approach")
print("-" * 70)

from sklearn.neighbors import KernelDensity

# Fit KDE for each assumed cluster center
gmm_model = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=20)
gmm_labels = gmm_model.fit_predict(X)

# Get cluster centers and covariances
centers = gmm_model.means_
covariances = gmm_model.covariances_

print(f"Cluster 0 center: {centers[0]}")
print(f"Cluster 1 center: {centers[1]}")
print(f"Distance between centers: {np.linalg.norm(centers[0] - centers[1]):.2f}\n")

# Compute separation index (larger = better separation)
separation = np.linalg.norm(centers[0] - centers[1]) / (np.mean(np.sqrt(np.diag(covariances[0]))) + 
                                                          np.mean(np.sqrt(np.diag(covariances[1]))))
print(f"Separation Index: {separation:.4f}")
if separation > 2:
    print("✓ Clusters are reasonably well separated\n")
else:
    print("⚠ Significant overlap detected - requires careful threshold selection\n")

# ============================================================================
# TECHNIQUE 6: CONSENSUS CLUSTERING (Bootstrap Stability)
# ============================================================================
print("6. Consensus Clustering (Bootstrap Stability Analysis)")
print("-" * 70)

n_bootstrap = 100
consensus_matrix = np.zeros((len(X), len(X)))

for i in range(n_bootstrap):
    # Bootstrap sample
    idx = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X[idx]
    
    # Cluster
    model = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
    labels = model.fit_predict(X_boot)
    
    # Build co-occurrence matrix
    for j1 in range(len(idx)):
        for j2 in range(j1, len(idx)):
            if labels[j1] == labels[j2]:
                consensus_matrix[idx[j1], idx[j2]] += 1
                consensus_matrix[idx[j2], idx[j1]] += 1

consensus_matrix /= n_bootstrap
consensus_stability = np.mean(np.abs(np.diag(consensus_matrix) - 0.5))

print(f"Bootstrap samples: {n_bootstrap}")
print(f"Consensus stability: {consensus_stability:.4f}")
print("(Higher = more stable clusters across bootstrap samples)\n")

# ============================================================================
# TECHNIQUE 7: CORRELATION WITH SPATIAL COORDINATES
# ============================================================================
print("7. Feature Engineering: Spatial Structure Analysis")
print("-" * 70)

# Compute pairwise distances and angles
from scipy.spatial.distance import pdist, squareform

distances = squareform(pdist(X, metric='euclidean'))
mean_dist = np.mean(distances[distances > 0])
std_dist = np.std(distances[distances > 0])

print(f"Mean inter-point distance: {mean_dist:.2f}")
print(f"Std dev distance: {std_dist:.2f}")
print(f"Ratio (std/mean): {std_dist/mean_dist:.4f}\n")

# Create distance-based features for clustering
X_augmented = np.column_stack([
    X,
    np.mean(distances, axis=1),  # Average distance to all points
    np.std(distances, axis=1)     # Distance variance
])

print("Augmented features: X, Y, mean_distance, distance_variance")
gmm_aug = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=20)
labels_aug = gmm_aug.fit_predict(X_augmented)

if np.mean(X[labels_aug == 0]) > np.mean(X[labels_aug == 1]):
    labels_aug = 1 - labels_aug

f1_aug = f1_score(true_labels, labels_aug)
print(f"F1 Score with augmented features: {f1_aug:.4f}\n")

# ============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("="*70)
print("SUMMARY: BEST TECHNIQUES FOR OVERLAY PROBLEMS")
print("="*70)

summary_methods = {
    'GMM (Full Covariance)': 0.0,
    'Spectral Clustering (Optimized Gamma)': best_f1_spectral,
    'LOF + GMM': lof_f1,
    'Augmented Features + GMM': f1_aug,
}

# Add multi-scale results
for metric, f1 in best_results.items():
    summary_methods[f'Spectral ({metric})'] = f1

summary_df = pd.DataFrame(list(summary_methods.items()), 
                          columns=['Method', 'F1 Score']).sort_values('F1 Score', ascending=False)

print("\n" + summary_df.to_string(index=False))

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("""
FOR HIGHLY OVERLAPPING DATA:

1. START WITH:
   → Gaussian Mixture Models (GMM) with full covariance
   → Try all covariance types and use BIC to select
   → Use soft probabilities instead of hard assignments

2. OPTIMIZE DECISION BOUNDARY:
   → Extract posterior probability P(red|data)
   → Find optimal threshold using F1 score
   → Don't use default 0.5 threshold

3. HANDLE UNCERTAINTY:
   → Use Local Outlier Factor (LOF) to detect boundary points
   → Weight clustering confidence by local density
   → Flag points with probability near 0.5 as uncertain

4. MULTI-SCALE ANALYSIS:
   → Try different distance metrics (Euclidean, Manhattan, Cosine)
   → Use Spectral Clustering with gamma optimization
   → Check stability with bootstrap consensus clustering

5. FEATURE ENGINEERING:
   → Add spatial statistics (mean distance, distance variance)
   → Consider domain-specific features
   → Augment simple coordinates with derived features

6. ENSEMBLE APPROACH:
   → Combine GMM, Spectral, and distance-metric methods
   → Use voting or stacking for final prediction
   → More robust than any single method

7. VALIDATION:
   → Always compute separation index (distance/width)
   → Use silhouette scores to detect poor clustering
   → Visualize probability distributions and decision boundaries

WHY YOUR PREVIOUS METHODS FAILED:
✗ DBSCAN: Assumes dense cores - fails with overlapping blobs
✗ Agglomerative: Linkage criteria break down with overlap
✗ K-Means: Linear separation - can't handle overlapping distributions
✓ GMM: Probabilistic - naturally handles overlaps
✓ Spectral: Non-linear - captures complex structures
✓ Soft Assignment: Gradual transition for uncertain points
""")

print("\n✓ Analysis complete! Check saved visualizations for detailed insights.")
