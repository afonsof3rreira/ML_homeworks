import itertools
import sys
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io.arff import loadarff
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os


def read_dataset(dataset_path: str):
    """A function to load a `.arff` dataset, skipping samples with at least a missing value for any variable.
    Args:
        dataset_path (str): The dataset full path.

    Returns:
        tuple: A tuple containing two np.ndarray as in (x_data, y_data), where x_data has (Nr samples x Nr features) dimensions and y_data has (Nr samples) dimensions.
    """
    # loading the dataset and converting into a np.ndarray
    dataset = loadarff(dataset_path)
    data = dataset[0]

    # tensor for counting conditional occurrences
    counter_tensor = np.zeros((9, 10, 2))  # dims (Vars, Values, conditioned to class c)

    # lists where to place x (feature values) and y data (ground-truth labels) for all samples
    x_data, y_data = [], []

    for sample in data:
        sample = np.array(list(sample)).astype('U13')
        sample_features = sample[:-1].astype(np.float32)

        class_ = sample[-1]
        if class_ == "benign":
            ind_c = 0
            ind_bool = False
        else:
            ind_c = 1
            ind_bool = True

        if not np.any(np.isnan(sample_features)):

            for i in range(sample_features.shape[0]):
                val = sample_features[i]
                counter_tensor[i, int(float(val)) - 1, ind_c] += 1

            x_data.append(np.asarray(sample_features).astype(int))
            y_data.append(ind_bool)

    # converting data into np.ndarray
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return (x_data, y_data)


def plot_boxplots(train_acc_max_feat, test_acc_max_feat, train_acc_max_depth, test_acc_max_depth, k_values, results_path, show_fig=True):
    # rearranging data to fit in a boxplot
    concat_data_feat = []
    concat_data_depth = []

    for train, test in zip(train_acc_max_feat, test_acc_max_feat):
        concat_data_feat.extend([train, test])

    for train, test in zip(train_acc_max_depth, test_acc_max_depth):
        concat_data_depth.extend([train, test])

    # creating x_ticks labels
    x_ticks_simple = ["K = {}".format(str(x)) for x in k_values]
    x_ticks_labels = list(
        itertools.chain.from_iterable([["{} \n train".format(x), "{} \n test".format(x)] for x in x_ticks_simple]))

    # plotting results
    fig_bp = plt.figure(figsize=plt.figaspect(0.5))
    ax1_bp = fig_bp.add_subplot(1, 2, 1)

    ax1_bp.boxplot(concat_data_feat, vert=True, patch_artist=True)

    # x-ticks settings
    ax1_bp.set_xticklabels(x_ticks_labels, Fontsize=10)

    # y-ticks settings
    major_ticks_y = np.arange(0.85, 1.0, 0.05)
    minor_ticks_y = np.arange(0.85, 1.0, 0.005)

    [ax1_bp.axvline(x + .5, color='k', alpha=0.4, linestyle='--') for x in ax1_bp.get_xticks()[1::2]]

    ax1_bp.set_yticks(major_ticks_y)
    ax1_bp.set_yticks(minor_ticks_y, minor=True)

    # grid settings
    ax1_bp.grid(which='both')
    ax1_bp.grid(which='minor', alpha=0.2)
    ax1_bp.grid(which='major', alpha=0.5)

    # labels, title and limits
    ax1_bp.set_ylabel('Accuracy')
    ax1_bp.set_title('$N_{feat.} = \{1, 3, 5, 9\}$')
    ax1_bp.set_ylim([0.85, 1.0025])

    ax2_bp = fig_bp.add_subplot(1, 2, 2)

    ax2_bp.boxplot(concat_data_depth, vert=True, patch_artist=True)

    # x-ticks settings
    ax2_bp.set_xticklabels(x_ticks_labels, Fontsize=10)

    # y-ticks settings
    major_ticks_y = np.arange(0.85, 1.0, 0.05)
    minor_ticks_y = np.arange(0.85, 1.0, 0.005)

    [ax2_bp.axvline(x + .5, color='k', alpha=0.4, linestyle='--') for x in ax2_bp.get_xticks()[1::2]]

    ax2_bp.set_yticks(major_ticks_y)
    ax2_bp.set_yticks(minor_ticks_y, minor=True)

    # grid settings
    ax2_bp.grid(which='both')
    ax2_bp.grid(which='minor', alpha=0.2)
    ax2_bp.grid(which='major', alpha=0.5)

    # labels, title and limits
    ax2_bp.set_title('$N_{depth} = \{1, 3, 5, 9\}$')
    ax2_bp.set_ylim([0.85, 1.0025])

    plt.suptitle('Training and Validation accuracies:')
    plt.savefig(os.path.join(results_path, 'accuracy_param_search'))
    if show_fig:
        plt.show()
    plt.close()

def plot_3D(train_acc_grid, test_acc_grid, k_values_1, k_values_2, results_path, show_fig=True):

    xpos = np.expand_dims(np.arange(train_acc_grid.shape[0]), axis=1)
    xpos = np.repeat(xpos, train_acc_grid.shape[1], axis=1)

    ypos = np.expand_dims(np.arange(train_acc_grid.shape[1]), axis=0)
    ypos = np.repeat(ypos, train_acc_grid.shape[0], axis=0)

    # TODO:
    input_data = train_acc_grid

    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    dx = 0.5
    dy = 0.5

    cmap = cm.get_cmap('jet')
    max_height = np.max(input_data.flatten())
    min_height = np.min(input_data.flatten())

    # scale each z to [0,1]
    rgba = [cmap((k - min_height) / np.abs(max_height - min_height)) for k in input_data.flatten()]

    dz = (input_data.flatten() - min_height)

    ax1.bar3d(xpos.flatten(), ypos.flatten(), min_height * np.ones(16, ), dx, dy, dz, color=rgba, zsort='average')
    ax1.set_zlim(min_height, max_height)
    ax1.set_xticks(np.arange(0.25, input_data.shape[0], 1))
    ax1.set_xticklabels(k_values_1)

    ax1.set_yticks(np.arange(0.25, input_data.shape[1], 1))
    ax1.set_yticklabels(k_values_2)

    ax1.set_zlabel("Accuracy")
    ax1.set_ylabel("$N_{depth}$")
    ax1.set_xlabel("$N_{feat.}$")
    ax1.set_title("Training")

    # TODO:
    input_data = test_acc_grid

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    dx = 0.5
    dy = 0.5

    cmap = cm.get_cmap('jet')
    max_height = np.max(input_data.flatten())
    min_height = np.min(input_data.flatten())

    # scale each z to [0,1]
    rgba = [cmap((k - min_height) / np.abs(max_height - min_height)) for k in input_data.flatten()]

    dz = (input_data.flatten() - min_height)

    ax2.bar3d(xpos.flatten(), ypos.flatten(), min_height * np.ones(16, ), dx, dy, dz, color=rgba, zsort='average')
    ax2.set_zlim(min_height, max_height)
    ax2.set_xticks(np.arange(0.25, input_data.shape[0], 1))
    ax2.set_xticklabels(k_values_1)

    ax2.set_yticks(np.arange(0.25, input_data.shape[1], 1))
    ax2.set_yticklabels(k_values_2)

    ax2.set_zlabel("Accuracy")
    ax2.set_ylabel("$N_{depth}$")
    ax2.set_xlabel("$N_{feat.}$")
    ax2.set_title("Validation")
    plt.suptitle("Decision tree accuracy vs ($N_{feat.}$, $N_{depth}$)")

    plt.savefig(os.path.join(results_path, "grid_3Dplot"))

    if show_fig:
        plt.show()
    plt.close()