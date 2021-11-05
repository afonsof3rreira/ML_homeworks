# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn import metrics
from sklearn.model_selection import learning_curve
import os

def read_dataset(dataset_path: str, feature_type='int', class_type='int', count_occurances=False):
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

    class_counts = [0, 0]

    for sample in data:
        sample = np.array(list(sample)).astype('U13')
        sample_features = sample[:-1].astype(np.float32)

        class_ = sample[-1]

        if class_type == 'int':

            if class_ == "benign":
                ind_c = 0
                ind_bool = False
                class_counts[0] += 1

            else:
                ind_c = 1
                ind_bool = True
                class_counts[1] += 1

        elif class_type == 'float':
            ind_bool = float(class_)

        if not np.any(np.isnan(sample_features)):

            if count_occurances:
                for i in range(sample_features.shape[0]):
                    val = sample_features[i]
                    counter_tensor[i, int(float(val)) - 1, ind_c] += 1

            x_data_temp = np.asarray(sample_features)

            if feature_type == 'int':
                x_data_temp = x_data_temp.astype(int)

            elif feature_type == 'float':
                x_data_temp = x_data_temp.astype(float)

            x_data.append(x_data_temp)
            y_data.append(ind_bool)

    # converting data into np.ndarray
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    print("Class counts for (benign, malign) are ({}, {})".format(class_counts[0], class_counts[1]))

    return (x_data, y_data)


def plot_boxplots(data, x_ticks, results_path, show_fig=True):

    # plotting results
    fig_bp = plt.figure(figsize=(7, 8))
    ax1_bp = fig_bp.add_subplot(1, 1, 1)

    ax1_bp.boxplot(data, vert=True, patch_artist=True)

    # x-ticks settings
    ax1_bp.set_xticklabels(x_ticks, Fontsize=10)

    data_arr = np.concatenate(data).ravel()

    min_ = min(np.min(data_arr), np.min(data_arr))
    max_ = max(np.max(data_arr), np.max(data_arr))

    max_ = max_ + 0.05 * abs(max_)
    min_ = min_ - 0.05 * abs(min_)

    lim_sup = max(abs(max_), abs(min_))

    # y-ticks settings
    major_ticks_y = np.linspace(-lim_sup, lim_sup, 5)
    minor_ticks_y = np.linspace(-lim_sup, lim_sup, 20)

    ax1_bp.set_yticks(major_ticks_y)
    ax1_bp.set_yticks(minor_ticks_y, minor=True)

    # grid settings
    ax1_bp.grid(which='both')
    ax1_bp.grid(which='minor', alpha=0.2)
    ax1_bp.grid(which='major', alpha=0.5)

    # labels, title and limits
    ax1_bp.set_ylabel('Residuals', fontsize=14)
    ax1_bp.set_ylim([-lim_sup, lim_sup])

    ax1_bp.set_xlabel('$\\alpha$', fontsize=14)

    plt.suptitle(
        'Statistical results of the residuals ($z - \hat{z}$) \n for MLP regression models with a varying hyperparameter $\\alpha$')
    plt.savefig(os.path.join(results_path, 'residuals_regularization'))

    if show_fig:
        plt.show()
    plt.close()


def plot_cm_comparison(cm_1, cm_2, results_path, filename, show_fig=True):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    cmd = metrics.ConfusionMatrixDisplay(cm_1, display_labels=['Benign', 'Malign'])
    cmd.plot(cmap='Blues', ax=axes[0])

    cmd_es = metrics.ConfusionMatrixDisplay(cm_2, display_labels=['Benign', 'Malign'])
    cmd_es.plot(cmap='Blues', ax=axes[1])

    axes[1].get_yaxis().set_visible(False)

    axes[0].set_title('Without Early Stopping')
    axes[1].set_title('With Early Stopping')

    fig.suptitle('Confusion matrix comparison for two MLP classification models', fontsize=16)

    plt.savefig(os.path.join(results_path, filename))

    if show_fig:
        plt.show()
    plt.close()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                            train_sizes=np.linspace(0.1, 1.0, 5), min_=None, max_=None):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if min_ is None:
            min_ = min(np.min(train_scores_mean), np.min(test_scores_mean))
        else:
            min_ = 0

        if max_ is None:
            max_ = max(np.max(train_scores_mean), np.max(test_scores_mean))
        else:
            max_ = 1.0

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        axes[0].set_ylim([min_, max_])

        return plt