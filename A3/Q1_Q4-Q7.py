# %% [markdown]
# # Question 1 a)

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# %%
k_values = [2,4,6,8]

data = pd.read_csv('/content/hw2_data.csv')
data


# %%
X = data.values[:, 1:]
labels_true = data.values[:, 0]

le = LabelEncoder()
labels_true = le.fit_transform(labels_true)

X_train, X_test, labels_train, labels_test = train_test_split(X, labels_true, test_size=0.2, random_state=0)

k_values = [2, 4, 6, 8]

ari_kmeans = np.zeros(len(k_values))
ari_gmm = np.zeros(len(k_values))

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans_labels_train = kmeans.fit_predict(X_train)
    kmeans_labels_test = kmeans.predict(X_test)
    ari_kmeans[k] = adjusted_rand_score(labels_test, kmeans_labels_test)

    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm_labels_train = gmm.fit_predict(X_train)
    gmm_labels_test = gmm.predict(X_test)
    ari_gmm[k] = adjusted_rand_score(labels_test, gmm_labels_test)


# %%
print("K-means ARI values:", ari_kmeans)
print("GMM ARI values:", ari_gmm)

# %% [markdown]
# Question 1b)
# K means performs significantly faster than GMM and is a lot less computationally intensive. It is able to correctly predict the cell type identity with fewer k-values and perfect accuracy wheras GMM needs more k-values. With more cell types, hard clustering methods like k means may not perform as well, but for this example it is ideal 

# %% [markdown]
# # Question 4

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
X = data.values[:, 1:]
labels_true = data.values[:, 0]

le = LabelEncoder()
labels_true = le.fit_transform(labels_true)

# Perform PCA with 15 components
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)

# Print the variance explained by each component
print("Variance explained by each component:")
print(pca.explained_variance_ratio_)

# Identify the most important component
most_important_component = np.argmax(pca.explained_variance_ratio_) + 1
print(f"The most important component is component {most_important_component}.")

# %%
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_true, cmap='viridis')
plt.xlabel("PC2")
plt.ylabel("PC1")
plt.title("Scatter plot of PC1 vs. PC2, colored by class label")
plt.show()

# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Extract the expression matrix and labels from the data
X = data.values[:, 1:]
labels = data.values[:, 0]

le = LabelEncoder()
labels_true = le.fit_transform(labels)

# Perform PCA with 15 components
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)

# Create colormap for class labels
num_classes = len(np.unique(labels_true))
cmap = ListedColormap(plt.cm.get_cmap('viridis', num_classes).colors)

# Create scatter plot of PC2 vs. PC3, colored by class label
plt.scatter(X_pca[:, 1], X_pca[:, 2], c=labels_true, cmap=cmap)

plt.xlabel("PC2")
plt.ylabel("PC3")
plt.title("Scatter plot of PC2 vs. PC3, colored by class label")
plt.show()

# %% [markdown]
# # Question 5

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# %%
import pandas as pd
import numpy as np
data = pd.read_csv('/content/A3RNAseq.csv', delimiter=',')
data


# %%
data

# %%
labels_data = pd.read_csv('/content/label.csv')
labels = labels_data['label']
labels = labels.T

# %%
# Perform PCA to reduce the dimensionality of the data to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform tSNE to create a 2D projection of the data
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Perform UMAP to create a 2D projection of the data
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2).fit_transform(X)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="jet")
plt.colorbar()
plt.title("PCA")

plt.subplot(1, 3, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="jet")
plt.colorbar()
plt.title("tSNE")

plt.subplot(1, 3, 3)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="jet")
plt.colorbar()
plt.title("UMAP")

plt.show()

# Calculate the Adjusted Rand Index (ARI)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
ari = adjusted_rand_score(labels, kmeans.labels_)
print("ARI:", ari)

# %%



