# imports
import sys

import graphviz
import numpy as np
from sklearn import tree, feature_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import os

from HW_2.utils.aux_functions import read_dataset, plot_boxplots, plot_3D

"""Homework 1 - programming solution (kNN and Naïve Bayes statistical analysis).

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

# creating a ./results/figures folder
results_figs_path = os.path.join(results_path, 'figures')
os.makedirs(results_figs_path, exist_ok=True)

(x_data, y_data) = read_dataset(dataset_path)

# splitting data ensuring consistent class balance
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)  # 59

# TODO: In a single plot, compare the training and testing accuracy of a decision tree with a varying:
# TODO: i. number of selected features in {1,3,5,9} using mutual information (tree with no fixed depth)

k_values = [1, 3, 5, 9]  # tested number of features
train_acc_max_feat, test_acc_max_feat = [], []

feat_inds = np.arange(0, 9, 1)

print("dataset size = {}".format(x_data.shape))
for k in k_values:

    print("Using K = {} neighbors".format(k))
    # print("splitting data into {} folds...".format(kf.get_n_splits(data)))
    i = 1
    train_acc_data_t = []
    test_acc_data_t = []
    feat_sel = feat_inds[:k]
    for train_index, test_index in kf.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        i += 1

        m_info = feature_selection.mutual_info_classif(x_train, y_train,
                                                       discrete_features=True,
                                                       copy=True, random_state=59)

        # going from mutual information scores -> feature indices
        m_info_inds = np.argsort(m_info)
        m_info_inds = m_info_inds[::-1]
        inds = m_info_inds[:k]

        x_train = x_train[:, inds]
        x_test = x_test[:, inds]

        dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None,
                                               random_state=59)  # for task i.
        # neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, weights='uniform')

        dec_tree.fit(x_train, y_train)

        train_pred = dec_tree.predict(x_train)
        test_pred = dec_tree.predict(x_test)

        train_acc_data_t.append(accuracy_score(y_train, train_pred))
        test_acc_data_t.append(accuracy_score(y_test, test_pred))

    # visualizing the generated tree for one of the k-folds only for simplicity
    dot_data = tree.export_graphviz(dec_tree, out_file=None,
                                    feature_names=np.asarray(labels_readable)[inds],
                                    class_names=['benign', 'malign'],
                                    filled=True, rounded=True,
                                    special_characters=True)

    scheme_path = os.path.join(results_figs_path, "test_i_{}_features".format(str(k)))
    graph = graphviz.Source(dot_data, filename=scheme_path)
    graph.render(filename=scheme_path, cleanup=True)

    train_acc_max_feat.append(np.asarray(train_acc_data_t))
    test_acc_max_feat.append(np.asarray(test_acc_data_t))

# TODO: ii. maximum tree depth in {1,3,5,9} (with all features and default parameters)
k_values_2 = [1, 3, 5, 9]  # tested number of features
train_acc_max_depth, test_acc_max_depth = [], []

print("dataset size = {}".format(x_data.shape))
for k in k_values_2:

    print("Using K = {} neighbors".format(k))
    # print("splitting data into {} folds...".format(kf.get_n_splits(data)))
    i = 1
    train_acc_data_t = []
    test_acc_data_t = []
    for train_index, test_index in kf.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        i += 1
        dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=k, random_state=59)    # for task ii.
        # neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, weights='uniform')

        dec_tree.fit(x_train, y_train)

        train_pred = dec_tree.predict(x_train)
        test_pred = dec_tree.predict(x_test)

        train_acc_data_t.append(accuracy_score(y_train, train_pred))
        test_acc_data_t.append(accuracy_score(y_test, test_pred))

    # visualizing the generated tree for one of the k-folds only for simplicity
    dot_data = tree.export_graphviz(dec_tree, out_file=None,
                                    feature_names=np.asarray(labels_readable)[inds],
                                    class_names=['benign', 'malign'],
                                    filled=True, rounded=True,
                                    special_characters=True)

    scheme_path = os.path.join(results_figs_path, "test_ii_{}_features".format(str(k)))
    graph = graphviz.Source(dot_data, filename=scheme_path)
    graph.render(filename=scheme_path, cleanup=True)

    train_acc_max_depth.append(np.asarray(train_acc_data_t))
    test_acc_max_depth.append(np.asarray(test_acc_data_t))

# plotting results
plot_boxplots(train_acc_max_feat, test_acc_max_feat,
              train_acc_max_depth, test_acc_max_depth,
              k_values, results_plots_path,
              show_fig=True)

# # TODO: (EXTRA) grid search for best (Max. tree depth, Max. Nr features) accuracy
train_acc_grid, test_acc_grid = [], []

for k1 in k_values:
    train_acc_sub, test_acc_sub = [], []
    for k2 in k_values_2:

        i = 1
        train_acc_data_t = []
        test_acc_data_t = []
        for train_index, test_index in kf.split(x_data, y_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            i += 1

            m_info = feature_selection.mutual_info_classif(x_train, y_train,
                                                           discrete_features=True,
                                                           copy=True, random_state=59)

            # going from mutual information scores -> feature indices
            m_info_inds = np.argsort(m_info)
            m_info_inds = m_info_inds[::-1]
            inds = m_info_inds[:k1]

            x_train = x_train[:, inds]
            x_test = x_test[:, inds]

            dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=k2, random_state=59)  # for task ii.
            # neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, weights='uniform')

            dec_tree.fit(x_train, y_train)

            train_pred = dec_tree.predict(x_train)
            test_pred = dec_tree.predict(x_test)

            train_acc_data_t.append(accuracy_score(y_train, train_pred))
            test_acc_data_t.append(accuracy_score(y_test, test_pred))

        train_acc_sub.append(np.mean(np.asarray(train_acc_data_t)))
        test_acc_sub.append(np.mean(np.asarray(test_acc_data_t)))

    train_acc_grid.append(train_acc_sub)
    test_acc_grid.append(test_acc_sub)

train_acc_grid = np.asarray(train_acc_grid)
test_acc_grid = np.asarray(test_acc_grid)

# plotting grid-search results
plot_3D(train_acc_grid, test_acc_grid,
        k_values, k_values_2,
        results_plots_path, show_fig=True)


