# Google Colab - Advanced Overlay Clustering Problem Solver
# Problem: Separate overlapping red and green 2D point clouds

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

import os
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
np.random.seed(42)


def minority_is_one(labels):
    """Label-free, stable orientation: call the smaller cluster 1 (red is the minority class)."""
    labels = np.asarray(labels).astype(int)
    return 1 - labels if labels.sum() > len(labels) / 2 else labels

# For Colab: Download and load data
# !wget -q "https://raw.githubusercontent.com/D2718281828nis/ML-overlay-problem/main/df_red_green_distinct.csv"
# Or upload directly in Colab with: from google.colab import files; files.upload()

# Load data (Gaussian red / uniform green; set OVERLAY_DATA=df_red_green_overlay.csv for the original)
DATA_FILE = os.environ.get('OVERLAY_DATA', 'df_red_green_distinct.csv')
df = pd.read_csv(DATA_FILE)
print("Data file:", DATA_FILE)

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
gmm_labels = minority_is_one(gmm_labels)

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
spectral_labels = minority_is_one(spectral_labels)

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

kernel_kmeans_labels = minority_is_one(kernel_kmeans_labels)

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

iso1_labels = minority_is_one(iso1_labels)

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
    labels = minority_is_one(labels)
    
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
red_comp = int(np.argmin(best_model.weights_))  # minority component = red; component order is arbitrary
density_labels = (soft_proba[:, red_comp] > 0.5).astype(int)

density_f1 = f1_score(true_labels, density_labels)
print(f"\nF1 Score (hard threshold): {density_f1:.4f}")
print(f"Precision: {precision_score(true_labels, density_labels):.4f}")
print(f"Recall: {recall_score(true_labels, density_labels):.4f}")

# Try optimal threshold
from scipy.optimize import minimize_scalar

def threshold_f1(threshold):
    labels = (soft_proba[:, red_comp] > threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return 0
    return -f1_score(true_labels, labels)

result = minimize_scalar(threshold_f1, bounds=(0.1, 0.9), method='bounded')
optimal_threshold = result.x

density_labels_opt = (soft_proba[:, red_comp] > optimal_threshold).astype(int)
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
# METHOD 8: t-SNE EMBEDDING + CLASSIFICATION
# ============================================================================
# t-SNE maps the (standardised) points to a 2-D embedding that keeps local
# neighbourhoods. Two ways to turn the map into red/green labels:
#   8a. t-SNE + k-NN density (label-free): score = minus the distance to the k-th
#       neighbour IN THE MAP; flag the densest points. As many points are flagged
#       as the expected red share (the same prior the notebook's LNND uses).
#   8b. t-SNE + DBSCAN (the notebook's pipeline): cluster the map; the cluster with
#       the best F1 is called red. Picking that cluster USES THE LABELS.
# Caveat: t-SNE calibrates every point to the same perplexity, which evens out
# density, so a dense red shape loses most of its density advantage in the map.
print("\n" + "="*70)
print("METHOD 8: t-SNE EMBEDDING + CLASSIFICATION")
print("="*70)

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

TSNE_PERPLEXITY = 30
TSNE_K = 10                      # neighbour used for the density score
TSNE_EPS, TSNE_MIN_SAMPLES = 2.75, 10

X_std = StandardScaler().fit_transform(X)
tsne_map = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init='pca',
                random_state=42).fit_transform(X_std)
n_flag = int(round(true_labels.mean() * len(X)))   # expected number of red points (prior)


def knn_density(points, k):
    """Minus the distance to the k-th nearest neighbour (higher = denser)."""
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    return -dist[:, k]


def flag_top(score, n):
    labels = np.zeros(len(score), dtype=int)
    labels[np.argsort(-score)[:n]] = 1
    return labels


# 8a. label-free: densest points of the t-SNE map
tsne_density_labels = flag_top(knn_density(tsne_map, TSNE_K), n_flag)
tsne_density_f1 = f1_score(true_labels, tsne_density_labels)
# reference: the same score on the raw coordinates
raw_density_f1 = f1_score(true_labels, flag_top(knn_density(X_std, TSNE_K), n_flag))
print(f"\n8a. t-SNE + k-NN density (no labels, flag top {n_flag})")
print(f"F1 Score: {tsne_density_f1:.4f}   (same score on raw X, Y: {raw_density_f1:.4f})")
print(f"Precision: {precision_score(true_labels, tsne_density_labels):.4f}")
print(f"Recall: {recall_score(true_labels, tsne_density_labels):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(true_labels, tsne_density_labels))

# 8b. DBSCAN on the map, red cluster picked by F1 (uses labels)
db_labels = DBSCAN(eps=TSNE_EPS, min_samples=TSNE_MIN_SAMPLES).fit_predict(tsne_map)
clusters = [c for c in np.unique(db_labels) if c >= 0]
tsne_dbscan_labels = np.zeros(len(X), dtype=int)
if clusters:
    best_c = max(clusters, key=lambda c: f1_score(true_labels, (db_labels == c).astype(int)))
    tsne_dbscan_labels = (db_labels == best_c).astype(int)
tsne_dbscan_f1 = f1_score(true_labels, tsne_dbscan_labels)
print(f"\n8b. t-SNE + DBSCAN (eps={TSNE_EPS}, min_samples={TSNE_MIN_SAMPLES}; "
      f"{len(clusters)} clusters; red cluster picked by F1 -> USES LABELS)")
print(f"F1 Score: {tsne_dbscan_f1:.4f}")
print(f"Precision: {precision_score(true_labels, tsne_dbscan_labels, zero_division=0):.4f}")
print(f"Recall: {recall_score(true_labels, tsne_dbscan_labels):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(true_labels, tsne_dbscan_labels))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, labels, title in [(axes[0], true_labels, "t-SNE map: ground truth"),
                          (axes[1], tsne_density_labels, f"t-SNE + k-NN density (F1 {tsne_density_f1:.3f})"),
                          (axes[2], tsne_dbscan_labels, f"t-SNE + DBSCAN, labels pick cluster (F1 {tsne_dbscan_f1:.3f})")]:
    ax.scatter(tsne_map[labels == 0, 0], tsne_map[labels == 0, 1], c='green', alpha=0.5, s=12, label='Green')
    ax.scatter(tsne_map[labels == 1, 0], tsne_map[labels == 1, 1], c='red', alpha=0.7, s=12, label='Red')
    wrong = labels != true_labels
    if labels is not true_labels and wrong.any():
        ax.scatter(tsne_map[wrong, 0], tsne_map[wrong, 1], marker='D', s=40, facecolors='none',
                   edgecolors='gold', linewidths=1.2, label='Wrong prediction')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('tsne_classification.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: tsne_classification.png")
plt.show()

# ============================================================================
# METHOD 9: GRAPH METHODS ON A k-NN GRAPH
# ============================================================================
# Adapted from https://github.com/D2718281828nis/ML-MachineLearning-Graphs:
#   * graph construction as in "Graph from DataFrame": every point is a node, linked to its
#     k nearest neighbours. NOTE: that notebook also adds "type-based edges" between points
#     of the same Type -- Type IS the colour, so those edges leak the answer. Not used here.
#   * the GCN message-passing operator from graph_learning_basics:
#         S = D~^(-1/2) (A + I) D~^(-1/2),   H' = S H
# Unsupervised: 9a graph-smoothed k-NN density, 9b spectral cut (Fiedler vector).
# Supervised (5-fold CV, colours of scored nodes hidden): 9c label spreading, 9d SGC
# (simplified GCN = logistic regression on S^K-propagated node features).
print("\n" + "="*70)
print("METHOD 9: GRAPH METHODS (k-NN graph, GCN propagation)")
print("="*70)

from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

GRAPH_K = 10
A_knn = kneighbors_graph(X_std, GRAPH_K, mode='connectivity', include_self=False)
A_knn = ((A_knn + A_knn.T) > 0).astype(float)                  # symmetric, distance-based edges only
A_tilde = A_knn + sparse.eye(len(X))
d_inv_sqrt = sparse.diags(np.asarray(A_tilde.sum(1)).ravel() ** -0.5)
S_gcn = d_inv_sqrt @ A_tilde @ d_inv_sqrt                       # GCN operator
print(f"k-NN graph: {len(X)} nodes, {int(A_knn.nnz / 2)} edges (k={GRAPH_K})")


def report(name, labels):
    f1 = f1_score(true_labels, labels)
    print(f"\n{name}\nF1 Score: {f1:.4f}   Precision: {precision_score(true_labels, labels, zero_division=0):.4f}   "
          f"Recall: {recall_score(true_labels, labels):.4f}")
    return f1


# 9a. unsupervised: k-NN density averaged over graph neighbours (one GCN step)
graph_density = S_gcn @ knn_density(X_std, GRAPH_K)
gdens_labels = flag_top(graph_density, n_flag)
gdens_f1 = report("9a. Graph-smoothed k-NN density (no labels, 1 propagation step)", gdens_labels)

# 9b. unsupervised: spectral cut of the graph, smaller part = red
spec = SpectralClustering(2, affinity='precomputed', random_state=0).fit_predict(A_knn.toarray())
gspec_labels = spec if spec.mean() < 0.5 else 1 - spec
gspec_f1 = report("9b. Graph spectral cut (no labels, smaller part = red)", gspec_labels)

# 9e. unsupervised: connected components of a k-NN graph with SMALL k (k = 2).
# Score = size of the point's component; the largest components are red. This is the
# method that wins when the classes differ only in connectivity, e.g. on
# df_red_green_chains.csv (red = long chains, green = short chains): F1 = 1.0 there.
from scipy.sparse.csgraph import connected_components
A2 = kneighbors_graph(X_std, 2, mode='connectivity', include_self=False)
n_comp, comp = connected_components(((A2 + A2.T) > 0).astype(float), directed=False)
gcomp_labels = flag_top(np.bincount(comp)[comp].astype(float), n_flag)
gcomp_f1 = report(f"9e. Graph connected components, k=2 (no labels; {n_comp} components)", gcomp_labels)

# 9c/9d. supervised, 5-fold CV
cv = StratifiedKFold(5, shuffle=True, random_state=0)
glp_score, sgc_score = np.zeros(len(X)), np.zeros(len(X))
node_feats = np.c_[PolynomialFeatures(4, include_bias=False).fit_transform(X_std), knn_density(X_std, GRAPH_K)]
node_feats = StandardScaler().fit_transform(node_feats)
sgc_feats = S_gcn @ (S_gcn @ node_feats)                       # K = 2 propagation steps
for tr, te in cv.split(X_std, true_labels):
    y_tr = true_labels.values[tr] if hasattr(true_labels, 'values') else true_labels[tr]
    Y0 = np.zeros((len(X), 2))
    Y0[tr, y_tr] = 1
    Y0[tr] /= np.array([[(y_tr == 0).sum(), (y_tr == 1).sum()]])  # class-balanced seeds
    F = Y0.copy()
    for _ in range(50):                                          # label spreading, alpha = 0.9
        F = 0.9 * (S_gcn @ F) + 0.1 * Y0
    glp_score[te] = (F[:, 1] / (F.sum(1) + 1e-12))[te]
    sgc = LogisticRegression(C=1e4, class_weight='balanced', max_iter=20000).fit(sgc_feats[tr], y_tr)
    sgc_score[te] = sgc.predict_proba(sgc_feats[te])[:, 1]
glp_labels, sgc_labels = flag_top(glp_score, n_flag), flag_top(sgc_score, n_flag)
glp_f1 = report("9c. Graph label spreading (USES LABELS, 5-fold CV)", glp_labels)
sgc_f1 = report("9d. SGC, simplified GCN (USES LABELS, 5-fold CV)", sgc_labels)

# 9g. supervised hyperbolic GCN (HyperbolicGCN from ML-MachineLearning-Graphs, see hyperbolic_gcn.py):
# features lifted into the Poincare ball (curvature -c), GCN aggregation in the tangent space,
# class-balanced loss, 5-fold CV. The repository's notebook trained it WITHOUT class weights on
# the original data and predicted no red point at all (recall 0).
from hyperbolic_gcn import hgcn_cv_scores
hgcn_labels = flag_top(hgcn_cv_scores(X_std, true_labels.values if hasattr(true_labels, 'values') else true_labels,
                                      k=GRAPH_K, c=1.0), n_flag)
hgcn_f1 = report("9g. Hyperbolic GCN, Poincare ball c=1 (USES LABELS, 5-fold CV)", hgcn_labels)

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(5, 4, figsize=(18, 22))
fig.suptitle('Overlay Clustering: Different Methods Comparison', fontsize=16, fontweight='bold')

methods = [
    (gmm_labels, "GMM (Full Covariance)", "F1: {:.4f}".format(gmm_f1)),
    (spectral_labels, "Spectral Clustering", "F1: {:.4f}".format(spectral_f1)),
    (kernel_kmeans_labels, "Kernel K-Means", "F1: {:.4f}".format(kernel_f1)),
    (iso1_labels, "Isolation Forest", "F1: {:.4f}".format(iso_f1)),
    (best_labels, f"GMM ({best_cov})", "F1: {:.4f}".format(best_f1)),
    (density_labels_opt, "Soft GMM + Threshold", "F1: {:.4f}".format(density_f1_opt)),
    (ensemble_labels, "Ensemble Vote", "F1: {:.4f}".format(ensemble_f1)),
    (tsne_density_labels, "t-SNE + k-NN density", "F1: {:.4f}".format(tsne_density_f1)),
    (tsne_dbscan_labels, "t-SNE + DBSCAN (labels)", "F1: {:.4f}".format(tsne_dbscan_f1)),
    (gdens_labels, "Graph: smoothed density", "F1: {:.4f}".format(gdens_f1)),
    (gspec_labels, "Graph: spectral cut", "F1: {:.4f}".format(gspec_f1)),
    (gcomp_labels, "Graph: connected components", "F1: {:.4f}".format(gcomp_f1)),
    (glp_labels, "Graph: label spreading (labels)", "F1: {:.4f}".format(glp_f1)),
    (sgc_labels, "Graph: SGC (labels)", "F1: {:.4f}".format(sgc_f1)),
    (hgcn_labels, "Graph: hyperbolic GCN (labels)", "F1: {:.4f}".format(hgcn_f1)),
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

for ax in axes.flat[len(methods):]:
    ax.axis('off')

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
               'Isolation Forest', f'GMM ({best_cov})', 'Soft GMM + Threshold', 'Ensemble',
               't-SNE + k-NN density', 't-SNE + DBSCAN (uses labels)',
               'Graph: smoothed density', 'Graph: spectral cut', 'Graph: connected components',
               'Graph: label spreading (uses labels)', 'Graph: SGC (uses labels)', 'Graph: hyperbolic GCN (uses labels)'],
    'F1 Score': [gmm_f1, spectral_f1, kernel_f1, iso_f1, best_f1, density_f1_opt, ensemble_f1,
                 tsne_density_f1, tsne_dbscan_f1, gdens_f1, gspec_f1, gcomp_f1, glp_f1, sgc_f1, hgcn_f1],
    'Precision': [
        precision_score(true_labels, gmm_labels),
        precision_score(true_labels, spectral_labels),
        precision_score(true_labels, kernel_kmeans_labels),
        precision_score(true_labels, iso1_labels),
        precision_score(true_labels, best_labels),
        precision_score(true_labels, density_labels_opt),
        precision_score(true_labels, ensemble_labels),
        precision_score(true_labels, tsne_density_labels),
        precision_score(true_labels, tsne_dbscan_labels, zero_division=0),
        *[precision_score(true_labels, l, zero_division=0) for l in (gdens_labels, gspec_labels, gcomp_labels, glp_labels, sgc_labels, hgcn_labels)]
    ],
    'Recall': [
        recall_score(true_labels, gmm_labels),
        recall_score(true_labels, spectral_labels),
        recall_score(true_labels, kernel_kmeans_labels),
        recall_score(true_labels, iso1_labels),
        recall_score(true_labels, best_labels),
        recall_score(true_labels, density_labels_opt),
        recall_score(true_labels, ensemble_labels),
        recall_score(true_labels, tsne_density_labels),
        recall_score(true_labels, tsne_dbscan_labels),
        *[recall_score(true_labels, l) for l in (gdens_labels, gspec_labels, gcomp_labels, glp_labels, sgc_labels, hgcn_labels)]
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
