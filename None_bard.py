import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier

# Import NumPy as np
import numpy as np

# Import manifold from sklearn
from sklearn import manifold

# Generate some data
data = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})

# Perform PCA
pca = PCA(2)
pca.fit(data)
x = pca.transform(data)

# Perform KMeans clustering
kmeans = KMeans(4, random_state=0)
kmeans.fit(x)
data['label'] = kmeans.predict(x)

# Plot the data
fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.scatter(x[:, 0][data['label'] == 0], x[:, 1][data['label'] == 0], c='blue', s=100, label='Cluster 0')
ax.scatter(x[:, 0][data['label'] == 1], x[:, 1][data['label'] == 1], c='red', s=100, label='Cluster 1')
ax.scatter(x[:, 0][data['label'] == 2], x[:, 1][data['label'] == 2], c='green', s=100, label='Cluster 2')
ax.scatter(x[:, 0][data['label'] == 3], x[:, 1][data['label'] == 3], c='orange', s=100, label='Cluster 3')
plt.legend()
plt.show()

# Perform TSNE
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
x_tsne = tsne.fit_transform(x)

# Plot the data
fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=data['label'], s=100, label='Clusters')
plt.legend()
plt.show()

# Create a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(x, data['label'])

# Visualize the decision tree
from sklearn import tree
plt.figure(figsize=(10, 10))
tree.plot_tree(clf, feature_names=data.columns, filled=True)
plt.show()
