import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error


"""Homework 2, part 1- programming solution (kNN and Naïve Bayes statistical analysis).

    Authors:
        - Afonso Ferreira - 86689
        - Rita Costa - 95968

"""

# defining paths where to retrieve the data from
script_dir = os.path.dirname(sys.argv[0])
filename = 'dataset_example_2.csv'
dataset_path = os.path.join(script_dir, 'data', filename)

df = pd.read_csv(dataset_path)
array = df.to_numpy()

x_data = array[:, :-1]
y_data = np.expand_dims(array[:, -1], axis=1)

# # train data
# x_train = x_data[:-2, :]
y_train = y_data[:-2, :]
#
# # test data
# x_test = x_data[-2:, :]
y_test = y_data[-2:, :]


basis_values = np.linalg.norm(x_data, ord=2, axis=1)

basis_values = np.repeat(np.expand_dims(basis_values, axis=1), x_data.shape[1]+1, axis=1)
power_matrix = np.repeat(np.expand_dims(np.arange(0, x_data .shape[1]+1), axis=0), x_data .shape[0], axis=0)

design_matrix = np.power(basis_values, power_matrix)

print("design matrix (trasnformation of the whole dataset):")
print(design_matrix)

x_train_t = design_matrix[:-2, :]
x_test_t = design_matrix[-2:, :]

design_term = np.matmul(np.transpose(x_train_t), x_train_t)
design_term = np.linalg.inv(design_term)

second_term = np.matmul(np.transpose(x_train_t), y_train)
weights = np.squeeze(np.matmul(design_term, second_term))

# y_train_pred = []
# for i in range(x_train.shape[0]):
#     x_set = np.ones((x_train.shape[1] + 1))
#     x_set[1:] = x_train[i]
#     y_train_pred.append(np.inner(x_set, weights))
#
# y_test_pred = []
# for i in range(x_test.shape[0]):
#     x_set = np.ones((x_test.shape[1] + 1))
#     x_set[1:] = x_test[i]
#     y_test_pred.append(np.inner(x_set, weights))

y_train_pred = []
for i in range(x_train_t.shape[0]):
    y_train_pred.append(np.inner(x_train_t[i, :], weights))

y_test_pred = []
for i in range(x_test_t.shape[0]):
    y_test_pred.append(np.inner(x_test_t[i, :], weights))


mse_train = mean_squared_error(y_train, np.asarray(y_train_pred), squared=False)  # squared=False to return the root
mse_test = mean_squared_error(y_test, np.asarray(y_test_pred), squared=False)  # squared=False to return the root

print("weights =")
print(weights)

print("mse train = {}".format(mse_train))
print("mse test = {}".format(mse_test))

# print(power_matrix)
#
# print(design_matrix)
# print(basis_values)