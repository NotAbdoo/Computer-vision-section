
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Generate virtual data points
np.random.seed(42)

points_1 = np.random.randn(50, 2) + [2, 2]
points_2 = np.random.randn(50, 2) + [7, 7]
points_3 = np.random.randn(50, 2) + [2, 8]

# Combine all points
X = np.vstack((points_1, points_2, points_3))

# 2. Apply K-Means Algorithm
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

# 3. Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Visualize the results
plt.figure(figsize=(10, 7))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='viridis',
    s=50,
    label='Data Points'
)

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    marker='X',
    s=250,
    edgecolors='black',
    label='Centroids'
)

plt.title("K-Means Clustering - Computer Vision")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# 5. Print results
print("Number of Clusters:", kmeans.n_clusters)
print("\nCentroids:")
print(centroids)

print("\nCluster Distribution:")
for i in range(3):
    print(f"Cluster {i}: {np.sum(labels == i)} points")
