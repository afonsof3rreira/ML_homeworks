# imports
import os
import sys
import warnings
import numpy as np
from matplotlib import MatplotlibDeprecationWarning
from sklearn import neural_network, metrics
from sklearn.model_selection import StratifiedKFold, KFold
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

# TODO: Using the breast.w.arff data from previous homeworks, show the confusion matrix of the
# TODO: aforementioned MLP in the presence and absence of early stopping.
print("Exercise 1 \n")

(x_data, y_data) = read_dataset(dataset_path)

# splitting data ensuring consistent class balance
splits = 5
kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=59)  # 59


print("dataset size = {}".format(x_data.shape[0]))

# print("splitting data into {} folds...".format(kf.get_n_splits(data)))
i = 1
train_acc_data_t = []
test_acc_data_t = []

y_true, y_pred = [], []
y_pred_es = []

y_true_train = []
y_pred_train = []
y_pred_es_train = []

for train_index, test_index in kf.split(x_data, y_data):
    print("Training MLP classification models using fold {}/{}...".format(i, splits))

    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    i += 1

    # model without Early Stopping
    neural_net_model = neural_network.MLPClassifier(hidden_layer_sizes=(3, 2), activation='relu', random_state=0,
                                                    validation_fraction=0.1, early_stopping=False)

    neural_net_model.fit(x_train, y_train)

    # model with Early Stopping
    neural_net_model_es = neural_network.MLPClassifier(hidden_layer_sizes=(3, 2), activation='relu', random_state=0,
                                                       validation_fraction=0.1, early_stopping=True)

    neural_net_model_es.fit(x_train, y_train)

    y_pred.append(neural_net_model.predict(x_test))

    y_true.append(y_test)
    y_pred_es.append(neural_net_model_es.predict(x_test))

    y_true_train.append(y_train)
    y_pred_train.append(neural_net_model.predict(x_train))
    y_pred_es_train.append(neural_net_model_es.predict(x_train))

print("Displaying Confusion matrix for the trained models...")

cm = metrics.confusion_matrix(np.concatenate(y_true).ravel(), np.concatenate(y_pred).ravel())
cm_es = metrics.confusion_matrix(np.concatenate(y_true).ravel(), np.concatenate(y_pred_es).ravel())

cm_train = metrics.confusion_matrix(np.concatenate(y_true_train).ravel(), np.concatenate(y_pred_train).ravel())
cm_es_train = metrics.confusion_matrix(np.concatenate(y_true_train).ravel(), np.concatenate(y_pred_es_train).ravel())

cm_acc = round(np.trace(cm) / np.sum(cm), 4)
cm_sensitivity = round(cm[0, 0] / np.sum(cm[0, :]), 4)
cm_specificity = round(cm[1, 1] / np.sum(cm[1, :]), 4)

cm_es_acc = round(np.trace(cm_es) / np.sum(cm_es), 4)
cm_es_sensitivity = round(cm_es[0, 0] / np.sum(cm_es[0, :]), 4)
cm_es_specificity = round(cm_es[1, 1] / np.sum(cm_es[1, :]), 4)


print("Validation results...")
print("accuracy = {}, sensitivity = {}, specificity = {}".format(cm_acc, cm_sensitivity, cm_specificity))
print("accuracy = {}, sensitivity = {}, specificity = {}".format(cm_es_acc, cm_es_sensitivity, cm_es_specificity))

cm_acc_train = round(np.trace(cm_train) / np.sum(cm_train), 4)
cm_sensitivity_train = round(cm_train[0, 0] / np.sum(cm_train[0, :]), 4)
cm_specificity_train = round(cm_train[1, 1] / np.sum(cm_train[1, :]), 4)

cm_es_acc_train = round(np.trace(cm_es_train) / np.sum(cm_es_train), 4)
cm_es_sensitivity_train = round(cm_es_train[0, 0] / np.sum(cm_es_train[0, :]), 4)
cm_es_specificity_train = round(cm_es_train[1, 1] / np.sum(cm_es_train[1, :]), 4)

print("Training results...")
print("accuracy = {}, sensitivity = {}, specificity = {}".format(cm_acc_train, cm_sensitivity_train, cm_specificity_train))
print("accuracy = {}, sensitivity = {}, specificity = {}".format(cm_es_acc_train, cm_es_sensitivity_train, cm_es_specificity_train))


plot_cm_comparison(cm, cm_es, results_plots_path, 'confusion_matrices_val', show_fig=True)
plot_cm_comparison(cm_train, cm_es_train, results_plots_path, 'confusion_matrices_train', show_fig=True)

print("\n")

# TODO: Using the kin8nm.arff, plot the distribution of the residues using boxplots in the presence
# TODO: and absence of regularization. Identify 4 strategies to minimize the observed error of the MLP
# TODO: regressor.

print("Exercise 2 \n")

filename_2 = 'kin8nm.arff'
dataset_path_2 = os.path.join(script_dir, 'data', filename_2)
(x_data, y_data) = read_dataset(dataset_path_2, feature_type='float', class_type='float', count_occurances=False)

# splitting data ensuring consistent class balance
splits = 5
kf = KFold(n_splits=splits, random_state=0, shuffle=True)

print("dataset size = {}".format(x_data.shape[0]))

# print("splitting data into {} folds...".format(kf.get_n_splits(data)))
i = 1
train_acc_data_t = []
test_acc_data_t = []

y_true = []

y_pred = [[], [], [], [], []]
reg_params = np.asarray([10, 1.0, 0.1, 0.0001, 0])

y_pred_es = []
y_pred_def = []
y_pred_1 = []

for train_index, test_index in kf.split(x_data, y_data):
    print("Training MLP regression models using fold {}/{}...".format(i, splits))

    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    i += 1

    # GT
    y_true.append(y_test)

    # alphas
    for reg_val, j in zip(reg_params, range(len(reg_params))):
        neural_net_model_temp = neural_network.MLPRegressor(hidden_layer_sizes=(3, 2), activation='relu', random_state=0,
                                                         validation_fraction=0.1, alpha=reg_val)

        neural_net_model_temp.fit(x_train, y_train)

        y_pred[j].append(neural_net_model_temp.predict(x_test))


print("Displaying Statistical Results of the predictions' residuals...")

# GT
y_true_vec = np.concatenate(y_true).ravel()

data = []

# alphas
for reg_val, j in zip(reg_params, range(len(reg_params))):

    y_pred_ = np.concatenate(y_pred[j]).ravel()
    res_reg = np.subtract(y_true_vec, y_pred_)
    data.append(res_reg)

x_ticks_simple = ['$10.0$', '$1.0$', '$0.1$', '$0.0001$', '$0.0$']
plot_boxplots(data, x_ticks_simple, results_plots_path, show_fig=True)