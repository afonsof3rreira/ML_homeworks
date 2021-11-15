# imports
import os
import sys
import warnings
import numpy as np
from matplotlib import MatplotlibDeprecationWarning
from sklearn import neural_network, metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from HW_3.utils.aux_functions import read_dataset, plot_boxplots, plot_cm_comparison
from sklearn.exceptions import ConvergenceWarning

# turning off warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

"""Homework 3 - programming solution.

    Authors:
        - Afonso Ferreira - 86689
        - Rita Costa - 95968

"""

# defining the labels as readable strings
labels_readable = ["Clump Thickness",
                   "Cell Size Uniformity",
                   "Cell Shape Uniformity",
                   "Marginal Adhesion",
                   "Single Epi Cell Size",
                   "Bare Nuclei",
                   "Bland Chromatin",
                   "Normal Nucleoli",
                   "Mitoses"]

# defining paths where to retrieve the data from
script_dir = os.path.dirname(sys.argv[0])
filename = 'breast.w.arff'
dataset_path = os.path.join(script_dir, 'data', filename)

# creating a ./results folder
results_path = os.path.join(script_dir, 'results')
os.makedirs(results_path, exist_ok=True)

# creating a ./results/plots folder
results_plots_path = os.path.join(results_path, 'plots')
os.makedirs(results_plots_path, exist_ok=True)

# TODO: Recall the breast.w.arff dataset from previous homeworks.
# TODO: Apply k-means clustering unsupervised on the original data with 𝑘 = 2 and 𝑘 = 3.
# TODO: a. Compare the produced solutions against the ECR (external measure)
# TODO: b. Compare the produced solutions against the Silhouette coefficient (internal measure).

print("Exercise 1 \n")

def ecr(y_data, y_pred):

    """Error Classification Rate (ECR) function for clustering.
        Args:
            y_data (np.ndarray): an array containing the true integer labels.
            y_pred (np.ndarray): an array containing the predicted integer labels.

        Returns:
            float: the ECR value
    """

    true_labels = np.unique(y_data)
    cluster_labels = np.unique(y_pred)

    n_clusters = cluster_labels.shape[0]
    ecr_val = 0

    for i in cluster_labels.tolist():
        # indices of samples assigned to cluster i
        samples_in_cluster_inds = np.argwhere(y_pred == i)

        # corresponding true labels
        samples_in_cluster = y_data[samples_in_cluster_inds]

        # vector containing counts
        in_cluster_counts = np.zeros(n_clusters)

        for j in true_labels.tolist():
            in_cluster_counts[j] = samples_in_cluster[samples_in_cluster == j].shape[0]

        ecr_val += (samples_in_cluster.shape[0] - np.max(in_cluster_counts))
        # for sample in

    return ecr_val / n_clusters

def clustering(y_data, y_pred):

    """Error Classification Rate (ECR) function for clustering.
        Args:
            y_data (np.ndarray): an array containing the true integer labels.
            y_pred (np.ndarray): an array containing the predicted integer labels.

        Returns:
            float: the ECR value
    """

    true_labels = np.unique(y_data)
    cluster_labels = np.unique(y_pred)

    x_data_ = []

    for i in cluster_labels.tolist():
        # indices of samples assigned to cluster i
        samples_in_cluster_inds = np.argwhere(y_pred == i)

        x_data_.append([np.squeeze(x_data[samples_in_cluster_inds]), i])

    x_data__ = []

    for j in true_labels.tolist():
        # indices of samples assigned to cluster i
        samples_true_inds = np.argwhere(y_data == j)

        x_data__.append([np.squeeze(x_data[samples_true_inds]), j])


    return (x_data_, x_data__)


(x_data, y_data) = read_dataset(dataset_path)

# converting boolean into int array \in [0, 1]
y_data = y_data.astype(int)

print("dataset size = {}".format(x_data.shape[0]))

# Applying clustering with K = 2
kmeans = KMeans(n_clusters=2, random_state=0).fit(x_data)
y_pred_2 = kmeans.predict(x_data)

ecr_2 = ecr(y_data, y_pred_2)
sil_2 = metrics.silhouette_score(x_data, y_pred_2)

print("Results for clustering with K = 2 clusters are: ECR = {}, Silhouette = {}".format(round(ecr_2, 4), round(sil_2, 4)))

# Applying clustering with K = 3
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_data)
y_pred_3 = kmeans.predict(x_data)

# getting the Centroids
centroids_3 = kmeans.cluster_centers_

ecr_3 = ecr(y_data, y_pred_3)
sil_3 = metrics.silhouette_score(x_data, y_pred_3)

print("Results for clustering with K = 3 clusters are: ECR = {}, Silhouette = {}".format(round(ecr_3, 4), round(sil_3, 4)))

print("\n")

# TODO: Visually plot the 𝑘 = 3 clustering solution using the top-2 features with higher mutual information.

feature_mi = mutual_info_classif(x_data, y_data, discrete_features=True)
feature_mi_inds = np.argsort(feature_mi)

feature_mi_inds = feature_mi_inds[::-1]
feature_mi = feature_mi[feature_mi_inds[:2]]

# corresponding labels
feature_mi_labels = [labels_readable[i] for i in feature_mi_inds]

# for K = 3
u_labels = np.unique(y_pred_3)

# plotting the results:
x_pred, x_true = clustering(y_data, y_pred_3)

ccode = ['C0', 'C1', 'C2']
# corresponding true labels



for i, j in zip(u_labels, range(len(u_labels))):

    samples_pred = x_pred[j][0]
    cluster = x_pred[j][1]
    sample_pred_vec = samples_pred[:, feature_mi_inds[:2]]

    centroids_temp = centroids_3[j, :]


    plt.scatter(sample_pred_vec[:, 0], sample_pred_vec[:, 1], label="$C_k = {}$".format(cluster), color=ccode[j])
    plt.scatter(centroids_temp[feature_mi_inds[0]], centroids_temp[feature_mi_inds[1]], s=250, label="Centroid of $C_k = {}$".format(cluster), color=ccode[j], alpha=.5)

line_styles = ['-', '--']
for k in np.unique(y_data).tolist():

    samples_true = x_true[k][0]
    labels_true = x_true[k][1]
    sample_true_vec = samples_true[:, feature_mi_inds[:2]]

    plt.scatter(sample_true_vec[:, 0], sample_true_vec[:, 1], label="True label = {}".format(labels_true), s=130, facecolors='none', edgecolors='k', marker='o', linestyle=line_styles[k])

plt.title("Clustering VS true labels")
plt.xlabel(feature_mi_labels[0])
plt.ylabel(feature_mi_labels[1])

plt.legend()
plt.show()