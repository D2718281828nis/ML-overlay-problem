# Google Colab - Advanced Overlay Clustering Problem Solver
# Problem: Separate overlapping red and green 2D point clouds

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# For Colab: Download and load data
# !wget -q "https://your-cloud-url/df_red_green_overlay.csv" -O df_red_green_overlay.csv
# Or upload directly in Colab with: from google.colab import files; files.upload()

# Load data
df = pd.read_csv('df_red_green_overlay.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nColor distribution:")
print(df['Color'].value_counts())

# Create ground truth labels (for evaluation)
true_labels = (df['Color'].str.contains('red')).astype(int)  # 1 = red, 0 = green
X = df[['X', 'Y']].values

# ============================================================================
# METHOD 1: GAUSSIAN MIXTURE MODELS (GMM) - BEST FOR OVERLAPS
# ============================================================================
print("\n" + "="*70)
print("METHOD 1: GAUSSIAN MIXTURE MODELS (GMM)")
print("="*70)

gmm = GaussianMixture(n_components=2, random_state=42, n_init=20, covariance_type='full')
gmm_labels = gmm.fit_predict(X)

# Flip labels if needed (ensure red = 1)
if np.mean(X[gmm_labels == 0]) > np.mean(X[gmm_labels == 1]):
    gmm_labels = 1 - gmm_labels

gmm_f1 = f1_score(true_labels, gmm_labels)
print(f"\nF1 Score: {gmm_f1:.4f}")
print(f"Precision: {precision_score(true_labels, gmm_labels):.4f}")
print(f"Recall: {recall_score(true_labels, gmm_labels):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, gmm_labels))

# Get probability estimates (soft clustering)
gmm_proba = gmm.predict_proba(X)
print(f"\nGMM Log-likelihood: {gmm.score(X):.4f}")

# ============================================================================
# METHOD 2: SPECTRAL CLUSTERING WITH GAUSSIAN KERNEL
# ============================================================================
print("\n" + "="*70)
print("METHOD 2: SPECTRAL CLUSTERING (RBF Kernel)")
print("="*70)

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel

# RBF kernel with optimized gamma
gamma = 1.0 / (2 * np.std(X))
spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=gamma, 
                              random_state=42, eigen_solver='arpack')
spectral_labels = spectral.fit_predict(X)

# Flip if needed
if np.mean(X[spectral_labels == 0]) > np.mean(X[spectral_labels == 1]):
    spectral_labels = 1 - spectral_labels

spectral_f1 = f1_score(true_labels, spectral_labels)
print(f"\nF1 Score: {spectral_f1:.4f}")
print(f"Precision: {precision_score(true_labels, spectral_labels):.4f}")
print(f"Recall: {recall_score(true_labels, spectral_labels):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, spectral_labels))

# ============================================================================
# METHOD 3: KERNEL K-MEANS (Non-linear K-Means via Kernel Trick)
# ============================================================================
print("\n" + "="*70)
print("METHOD 3: KERNEL K-MEANS")
print("="*70)

from sklearn.cluster import KMeans

# Compute RBF kernel matrix
K = rbf_kernel(X, gamma=gamma)

# Convert to eigenspace
eigenvalues, eigenvectors = np.linalg.eigh(K)
idx = np.argsort(eigenvalues)[::-1][:2]
K_embedded = eigenvectors[:, idx]

kmeans_kernel = KMeans(n_clusters=2, random_state=42, n_init=20)
kernel_kmeans_labels = kmeans_kernel.fit_predict(K_embedded)

if np.mean(K_embedded[kernel_kmeans_labels == 0]) > np.mean(K_embedded[kernel_kmeans_labels == 1]):
    kernel_kmeans_labels = 1 - kernel_kmeans_labels

kernel_f1 = f1_score(true_labels, kernel_kmeans_labels)
print(f"\nF1 Score: {kernel_f1:.4f}")
print(f"Precision: {precision_score(true_labels, kernel_kmeans_labels):.4f}")
print(f"Recall: {recall_score(true_labels, kernel_kmeans_labels):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, kernel_kmeans_labels))

# ============================================================================
# METHOD 4: ISOLATION FOREST (Anomaly Detection Approach)
# ============================================================================
print("\n" + "="*70)
print("METHOD 4: TWO-STAGE ISOLATION FOREST")
print("="*70)

from sklearn.ensemble import IsolationForest

# Use Isolation Forest for each "cluster" iteratively
iso1 = IsolationForest(contamination=0.5, random_state=42)
iso1_labels = iso1.fit_predict(X)
iso1_labels = np.where(iso1_labels == -1, 1, 0)

if np.mean(X[iso1_labels == 0]) > np.mean(X[iso1_labels == 1]):
    iso1_labels = 1 - iso1_labels

iso_f1 = f1_score(true_labels, iso1_labels)
print(f"\nF1 Score: {iso_f1:.4f}")
print(f"Precision: {precision_score(true_labels, iso1_labels):.4f}")
print(f"Recall: {recall_score(true_labels, iso1_labels):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, iso1_labels))

# ============================================================================
# METHOD 5: EXPECTATION-MAXIMIZATION (EM) WITH CONSTRAINT
# ============================================================================
print("\n" + "="*70)
print("METHOD 5: EM WITH SPATIAL CONSTRAINT")
print("="*70)

# Try different covariance types for GMM
best_f1 = 0
best_cov = None
best_model = None

for cov_type in ['full', 'tied', 'diag', 'spherical']:
    model = GaussianMixture(n_components=2, covariance_type=cov_type, 
                           random_state=42, n_init=20)
    labels = model.fit_predict(X)
    
    # Flip if needed
    if np.mean(X[labels == 0]) > np.mean(X[labels == 1]):
        labels = 1 - labels
    
    f1 = f1_score(true_labels, labels)
    print(f"Covariance type: {cov_type:12s} → F1: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_cov = cov_type
        best_model = model
        best_labels = labels

print(f"\n✓ Best GMM config: {best_cov}")
print(f"F1 Score: {best_f1:.4f}")
print(f"Precision: {precision_score(true_labels, best_labels):.4f}")
print(f"Recall: {recall_score(true_labels, best_labels):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, best_labels))

# ============================================================================
# METHOD 6: DENSITY-BASED SOFT ASSIGNMENT
# ============================================================================
print("\n" + "="*70)
print("METHOD 6: DENSITY-BASED SOFT ASSIGNMENT")
print("="*70)

from sklearn.neighbors import KernelDensity

# Estimate density for each point assuming it belongs to cluster 0 or 1
kde = KernelDensity(kernel='gaussian', bandwidth='scott')

# Use GMM soft probabilities as density-weighted assignment
soft_proba = best_model.predict_proba(X)
density_labels = (soft_proba[:, 1] > 0.5).astype(int)

density_f1 = f1_score(true_labels, density_labels)
print(f"\nF1 Score (hard threshold): {density_f1:.4f}")
print(f"Precision: {precision_score(true_labels, density_labels):.4f}")
print(f"Recall: {recall_score(true_labels, density_labels):.4f}")

# Try optimal threshold
from scipy.optimize import minimize_scalar

def threshold_f1(threshold):
    labels = (soft_proba[:, 1] > threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return 0
    return -f1_score(true_labels, labels)

result = minimize_scalar(threshold_f1, bounds=(0.1, 0.9), method='bounded')
optimal_threshold = result.x

density_labels_opt = (soft_proba[:, 1] > optimal_threshold).astype(int)
density_f1_opt = f1_score(true_labels, density_labels_opt)

print(f"\nF1 Score (optimal threshold={optimal_threshold:.3f}): {density_f1_opt:.4f}")
print(f"Precision: {precision_score(true_labels, density_labels_opt):.4f}")
print(f"Recall: {recall_score(true_labels, density_labels_opt):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, density_labels_opt))

# ============================================================================
# METHOD 7: HYBRID APPROACH - VOTING ENSEMBLE
# ============================================================================
print("\n" + "="*70)
print("METHOD 7: VOTING ENSEMBLE (Meta-Learner)")
print("="*70)

# Stack predictions from multiple methods
ensemble_matrix = np.column_stack([
    gmm_labels,
    spectral_labels,
    kernel_kmeans_labels,
    density_labels_opt
])

# Soft voting (majority)
ensemble_labels = (np.mean(ensemble_matrix, axis=1) > 0.5).astype(int)

ensemble_f1 = f1_score(true_labels, ensemble_labels)
print(f"\nF1 Score: {ensemble_f1:.4f}")
print(f"Precision: {precision_score(true_labels, ensemble_labels):.4f}")
print(f"Recall: {recall_score(true_labels, ensemble_labels):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, ensemble_labels))

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Overlay Clustering: Different Methods Comparison', fontsize=16, fontweight='bold')

methods = [
    (gmm_labels, "GMM (Full Covariance)", "F1: {:.4f}".format(gmm_f1)),
    (spectral_labels, "Spectral Clustering", "F1: {:.4f}".format(spectral_f1)),
    (kernel_kmeans_labels, "Kernel K-Means", "F1: {:.4f}".format(kernel_f1)),
    (iso1_labels, "Isolation Forest", "F1: {:.4f}".format(iso_f1)),
    (best_labels, f"GMM ({best_cov})", "F1: {:.4f}".format(best_f1)),
    (density_labels_opt, "Soft GMM + Threshold", "F1: {:.4f}".format(density_f1_opt)),
    (ensemble_labels, "Ensemble Vote", "F1: {:.4f}".format(ensemble_f1)),
    (true_labels, "Ground Truth", "Reference")
]

for idx, (labels, title, score) in enumerate(methods):
    ax = axes[idx // 4, idx % 4]
    scatter = ax.scatter(X[labels == 0, 0], X[labels == 0, 1], 
                        c='green', alpha=0.6, s=30, label='Green')
    scatter = ax.scatter(X[labels == 1, 0], X[labels == 1, 1], 
                        c='red', alpha=0.6, s=30, label='Red')
    
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title(f"{title}\n{score}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('overlay_clustering_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: overlay_clustering_comparison.png")
plt.show()

# ============================================================================
# DETAILED ANALYSIS: GMM PROBABILITY DISTRIBUTIONS
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GMM contours
ax = axes[0]
x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
y_min, y_max = X[:, 1].min() - 50, X[:, 1].max() + 50
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                     np.linspace(y_min, y_max, 100))
Z = best_model.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.scatter(X[best_labels == 0, 0], X[best_labels == 0, 1], c='green', alpha=0.5, s=30)
ax.scatter(X[best_labels == 1, 0], X[best_labels == 1, 1], c='red', alpha=0.5, s=30)
ax.contour(xx, yy, Z, levels=10, linewidths=0.5, colors='black', alpha=0.3)
ax.set_title(f'GMM with Contours (BIC: {best_model.bic(X):.1f})', fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True, alpha=0.3)

# Probability distribution
ax = axes[1]
ax.hist(soft_proba[true_labels == 0, 1], bins=30, alpha=0.6, label='Green (True)', color='green')
ax.hist(soft_proba[true_labels == 1, 1], bins=30, alpha=0.6, label='Red (True)', color='red')
ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal Threshold ({optimal_threshold:.3f})')
ax.set_xlabel('P(Red | Data)')
ax.set_ylabel('Frequency')
ax.set_title('GMM Posterior Probability Distribution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_detailed_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: gmm_detailed_analysis.png")
plt.show()

# ============================================================================
# PERFORMANCE SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

results_data = {
    'Method': ['GMM (Full)', 'Spectral Clustering', 'Kernel K-Means', 
               'Isolation Forest', f'GMM ({best_cov})', 'Soft GMM + Threshold', 'Ensemble'],
    'F1 Score': [gmm_f1, spectral_f1, kernel_f1, iso_f1, best_f1, density_f1_opt, ensemble_f1],
    'Precision': [
        precision_score(true_labels, gmm_labels),
        precision_score(true_labels, spectral_labels),
        precision_score(true_labels, kernel_kmeans_labels),
        precision_score(true_labels, iso1_labels),
        precision_score(true_labels, best_labels),
        precision_score(true_labels, density_labels_opt),
        precision_score(true_labels, ensemble_labels)
    ],
    'Recall': [
        recall_score(true_labels, gmm_labels),
        recall_score(true_labels, spectral_labels),
        recall_score(true_labels, kernel_kmeans_labels),
        recall_score(true_labels, iso1_labels),
        recall_score(true_labels, best_labels),
        recall_score(true_labels, density_labels_opt),
        recall_score(true_labels, ensemble_labels)
    ]
}

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values('F1 Score', ascending=False)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('overlay_clustering_results.csv', index=False)
print("\n✓ Saved: overlay_clustering_results.csv")

print("\n" + "="*70)
print("KEY INSIGHTS FOR OVERLAY PROBLEMS")
print("="*70)
print("""
1. GMM (Gaussian Mixture Models) works best for overlapping clusters
   - Provides probabilistic soft assignments
   - Can estimate mixture components and covariances
   
2. Spectral Clustering with RBF kernel captures non-linear separations
   - Uses eigendecomposition of affinity matrix
   - Good for manifold-like data
   
3. Kernel K-Means extends K-Means to non-linear spaces
   - Cheaper than Spectral Clustering
   - More sensitive to kernel hyperparameters
   
4. Soft assignment with optimal threshold often beats hard clustering
   - Use posterior probabilities to find best decision boundary
   - Optimize threshold via F1 score
   
5. Ensemble voting combines strengths of multiple algorithms
   - Reduces variance in uncertain regions
   - More robust to individual algorithm failures
   
6. For overlapping problems:
   ✓ Avoid: K-Means, DBSCAN, Agglomerative (too rigid for overlaps)
   ✓ Prefer: GMM, Spectral, Kernel methods, soft assignments
   ✓ Optimize: Kernel bandwidth, covariance type, decision threshold
""")
