import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Example input data (nested list)
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

# PCA to choose most relevant features
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(data)

# t-SNE to choose most relevant features
tsne = TSNE(n_components=2)
reduced_data_tsne = tsne.fit_transform(data)

# KMeans clustering
kmeans = KMeans(n_clusters=2)
cluster_labels = kmeans.fit_predict(data)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(data, cluster_labels)

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(data, cluster_labels)

# Getting explanation from Decision Tree
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
target_names = ['Cluster 1', 'Cluster 2']
tree_rules = decision_tree.tree_

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_names = [
        feature_names[i] if i != -2 else "output"
        for i in tree_.feature
    ]
    paths = []
    path = []
    
    def recurse(node, path):
        if tree_.feature[node] != -2:
            name = feature_names[node]
            threshold = tree_.threshold[node]
            path.append(f"if {name} <= {threshold}:")
            recurse(tree_.children_left[node], path)
            path[-1] = path[-1][:-1] + f" and {name} > {threshold}:"
            recurse(tree_.children_right[node], path)
            path.pop()
        else:
            path.append(f"return {np.argmax(tree_.value[node])}")
            paths.append(" ".join(path))
            path.pop()
    
    recurse(0, path)
    return paths

decision_tree_explanation = tree_to_code(decision_tree, feature_names)

# Print the explanations
print("PCA reduced data:")
print(reduced_data_pca)
print()

print("t-SNE reduced data:")
print(reduced_data_tsne)
print()

print("KMeans clustering labels:")
print(cluster_labels)
print()

print("Decision Tree explanations:")
for explanation in decision_tree_explanation:
    print(explanation)
print()
