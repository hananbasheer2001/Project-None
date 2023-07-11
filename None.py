# Importing the required libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Feature selection using PCA or TSNE
def feature_selection(data):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    # Alternatively, perform t-SNE
    # tsne = TSNE(n_components=2)
    # tsne_result = tsne.fit_transform(data)
    
    return pca_result

# Clustering using KNN or K-Means
def perform_clustering(data):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(data)
    # Alternatively, perform KNN clustering
    # knn = KNN(n_clusters=3)
    # cluster_labels = knn.fit_predict(data)
    
    return cluster_labels

# Describing logic using Decision Tree or Random Forest
def describe_logic(data, labels):
    # Fit a Decision Tree classifier
    dt = DecisionTreeClassifier()
    dt.fit(data, labels)
    # Alternatively, fit a Random Forest classifier
    # rf = RandomForestClassifier()
    # rf.fit(data, labels)
    
    # Extract the rules/logic from the trained model
    logic = dt.tree_.feature
    
    return logic

# Generate a sentence using the logic
def generate_sentence(logic):
    sentence = "The most relevant features were selected using PCA. The data was then clustered using K-Means. The logic for the clusters was described using a Decision Tree."
    # Alternatively, if Random Forest was used:
    # sentence = "The most relevant features were selected using t-SNE. The data was then clustered using KNN. The logic for the clusters was described using a Random Forest."
    
    return sentence

# Example usage
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example input data
selected_features = feature_selection(data)
cluster_labels = perform_clustering(selected_features)
logic = describe_logic(selected_features, cluster_labels)
sentence = generate_sentence(logic)
print("Data:", data)
print("Selected features:", selected_features)
print("Cluster labels:", cluster_labels)
print("Logic:", logic)









