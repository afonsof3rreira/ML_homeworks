# imports
import itertools
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os
"""Homework 1 programming resolution.

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

# loading the dataset and converting into a np.ndarray
dataset = loadarff(dataset_path)
data = dataset[0]

# tensor for counting conditional occurrences
counter_tensor = np.zeros((9, 10, 2))  # dims (Vars, Values, conditioned to class c)

# lists where to place x (feature values) and y data (ground-truth labels) for all samples
x_data, y_data = [], []
z = 0
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

    add_sample = True

    if not np.any(np.isnan(sample_features)):
        for i in range(sample_features.shape[0]):
            val = sample_features[i]
            counter_tensor[i, int(float(val)) - 1, ind_c] += 1
    else:
        add_sample = False

    if add_sample:
        x_data.append(np.asarray(sample_features).astype(int))
        y_data.append(ind_bool)
    z += 1

# getting conditioned probabilities from counts
total_counts = np.sum(counter_tensor, axis=1)
total_counts = np.expand_dims(total_counts, axis=1)
total_counts = np.repeat(total_counts, 10, axis=1)
probs = np.divide(counter_tensor, total_counts)

# TODO: II.5) Resolution: Draw the class-conditional distributions per variable using a 3x3 plot grid
fig = plt.figure()
ax = []
x = np.arange(1, 11, 1)

for i in range(1, 10):

    ax.append(fig.add_subplot(3, 3, i))

    plt.plot(x, probs[i - 1, :, 0], label="benign")
    plt.plot(x, probs[i - 1, :, 1], label="malign")
    plt.legend()

    plt.title(labels_readable[i - 1])

    plt.ylim([0, 1])

    plt.yticks(np.arange(0, 1.01, 0.2))
    ax[i - 1].set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    ax[i - 1].grid(which='minor', alpha=0.2)
    ax[i - 1].grid(which='major', alpha=0.5)
    ax[i - 1].autoscale(enable=True, axis='x', tight=True)

    if i not in [1, 4, 7]:
        ax[i - 1].yaxis.set_ticklabels([])
        ax[i - 1].tick_params(axis='y', which='both', length=0)

    if i not in [7, 8, 9]:
        ax[i - 1].xaxis.set_ticklabels([])
        ax[i - 1].tick_params(axis='x', which='both', length=0)

    xticks = np.arange(1, 11, 1)
    plt.xticks(xticks)

fig.text(0.5, 0.04, 'Feature value', ha='center')
fig.text(0.04, 0.5, 'Conditional probability', va='center', rotation='vertical')
plt.show()
plt.close()


# TODO: II.6) Resolution: Using a 10-fold cross validation with seed=<group number>, assess the accuracy of 𝑘NN
# TODO: under 𝑘 ∈ {3,5,7}, Euclidean distance and uniform weights. Show empirically, which 𝑘 is less
# TODO: susceptible to the overfitting risk?

# converting data into np.ndarray
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

# splitting data ensuring consistent class balance
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)  # 59

k_values = [1, 3, 5, 7, 9, 11]  # tested K values for kNN
train_acc_data, test_acc_data = [], []

print("dataset size = {}".format(x_data.shape))
for k in k_values:

    print("Using K = {} neighbors".format(k))
    # print("splitting data into {} folds...".format(kf.get_n_splits(data)))
    i = 1
    train_acc_data_t = []
    test_acc_data_t = []
    for train_index, test_index in kf.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        i += 1
        neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, weights='uniform')

        neigh.fit(x_train, y_train)
        train_pred = neigh.predict(x_train)
        test_pred = neigh.predict(x_test)

        train_acc_data_t.append(accuracy_score(y_train, train_pred))
        test_acc_data_t.append(accuracy_score(y_test, test_pred))

    train_acc_data.append(train_acc_data_t)
    test_acc_data.append(test_acc_data_t)

# rearranging data to fit in a boxplot
concat_data = []

for train, test in zip(train_acc_data, test_acc_data):
    concat_data.append(train)
    concat_data.append(test)

# creating x_ticks labels
x_ticks_simple = ["K = {}".format(str(x)) for x in k_values]
x_ticks_labels = list(
    itertools.chain.from_iterable([["{} \n train".format(x), "{} \n test".format(x)] for x in x_ticks_simple]))

# plotting results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.boxplot(concat_data, vert=True, patch_artist=True)

# x-ticks settings
ax.set_xticklabels(x_ticks_labels, Fontsize=10)

# y-ticks settings
major_ticks_y = np.arange(0.85, 1.0, 0.05)
minor_ticks_y = np.arange(0.85, 1.0, 0.005)

[ax.axvline(x + .5, color='k', alpha=0.4, linestyle='--') for x in ax.get_xticks()[1::2]]

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# grid settings
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

# labels, title and limits
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy for K={1, 3, 5, 7, 9, 11} neighbors')
plt.ylim([0.85, 1.0025])
plt.show()
plt.close()


# TODO: II.7) Resolution: Fixing 𝑘 = 3, and assuming accuracy estimates are normally distributed, test the
# TODO: hypothesis “𝑘NN is statistically superior to Naïve Bayes (multinomial assumption)”

# computing accuracy and F1-score for kNN with K=3 and Naïve Bayes
k = 3
print("Using K = {} neighbors".format(k))
# print("splitting data into {} folds...".format(kf.get_n_splits(data)))
i = 1
acc_compare = [[], [], [], []]
f1_score_compare = [[], [], [], []]
for train_index, test_index in kf.split(x_data, y_data):
    print("computing {}...".format(str(i)))

    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    i += 1
    neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, weights='uniform')

    neigh.fit(x_train, y_train)
    train_pred_knn = neigh.predict(x_train)
    test_pred_knn = neigh.predict(x_test)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    train_pred_bayes = gnb.predict(x_train)
    test_pred_bayes = gnb.predict(x_test)

    acc_compare[0].append(accuracy_score(y_train, train_pred_knn))
    acc_compare[1].append(accuracy_score(y_test, test_pred_knn))
    acc_compare[2].append(accuracy_score(y_train, train_pred_bayes))
    acc_compare[3].append(accuracy_score(y_test, test_pred_bayes))

    f1_score_compare[0].append(f1_score(y_train, train_pred_knn))
    f1_score_compare[1].append(f1_score(y_test, test_pred_knn))
    f1_score_compare[2].append(f1_score(y_train, train_pred_bayes))
    f1_score_compare[3].append(f1_score(y_test, test_pred_bayes))

# plotting accuracy comparison results
fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
plt.boxplot(acc_compare, vert=True, patch_artist=True)
ax.set_xticklabels(['train kNN', 'test kNN', 'train NB', 'test NB'], Fontsize=10)

major_ticks_y = np.arange(0.85, 1.0, 0.05)
minor_ticks_y = np.arange(0.85, 1.0, 0.005)

[ax.axvline(x + .5, color='k', alpha=0.4, linestyle='--') for x in ax.get_xticks()[1::2]]

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy: kNN vs Naïve Bayes (NB)')
plt.ylim([0.85, 1.0025])

plt.show()
plt.close()

# plotting F1-score comparison results
fig3 = plt.figure()
ax = fig3.add_subplot(1, 1, 1)
plt.boxplot(f1_score_compare, vert=True, patch_artist=True)
ax.set_xticklabels(['train kNN', 'test kNN', 'train NB', 'test NB'], Fontsize=10)

major_ticks_y = np.arange(0.85, 1.0, 0.05)
minor_ticks_y = np.arange(0.85, 1.0, 0.005)

[ax.axvline(x + .5, color='k', alpha=0.4, linestyle='--') for x in ax.get_xticks()[1::2]]

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.ylabel('F1-score')
plt.title('Training and Validation F1-Score: kNN vs Naïve Bayes (NB)')
plt.ylim([0.85, 1.0025])

plt.show()
plt.close()
