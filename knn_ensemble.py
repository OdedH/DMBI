__author__ = 'Oded  Hupert'

from sklearn import svm
from sklearn import grid_search
from sklearn import feature_selection
from sklearn import preprocessing
import csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier

# import from Excel
with open('Dataset_C_transposed_reduced.csv', 'rb') as f:
    reader = csv.reader(f)
    raw_data = list(reader)

target_names = raw_data[0]
target = []
data = []
for row in raw_data[1:]:
    target.append(row[0])
    data.append(map(lambda x: float(x), row[1:]))


random_num = random.sample(xrange(7070), 100)
random_data = []
for row in raw_data[1:]:
    new_raw = []
    for i in row[1:]:
        if i in random_num:
            new_raw.append(row[1:][i])
    random_data.append(map(lambda x: float(x), new_raw[1:]))

# scale data for SVM
scaled_data = (preprocessing.MinMaxScaler()).fit_transform(data)
# feature selection

X_train = []
y_train = []
for i in range(len(scaled_data)):
    if i not in [5, 8, 24, 28, 42, 54]:
        X_train.append(scaled_data[i])
        y_train.append(target[i])

X_test = list(scaled_data[i] for i in [5, 8, 24, 28, 42, 54])
y_test = list(target[i] for i in [5, 8, 24, 28, 42, 54])

nbrs = KNeighborsClassifier(n_neighbors=7)
nbrs.fit(X_train, y_train)

result = nbrs.predict(X_test)
print result
print y_test
