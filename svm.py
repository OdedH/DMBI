__author__ = 'Oded  Hupert'

from sklearn import svm
from sklearn import grid_search
from sklearn import feature_selection
from sklearn import preprocessing
import csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

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

# scale data for SVM
scaled_data = (preprocessing.MinMaxScaler()).fit_transform(data)
# feature selection
feature_selecting_chi = feature_selection.SelectKBest(feature_selection.chi2, k=100).fit(scaled_data, target)
feature_selected = feature_selecting_chi.get_support(True)
feature_selected_names = map(lambda x: target_names[x + 1], feature_selected)
data_selected = feature_selection.SelectKBest(feature_selection.chi2, k=100).fit_transform(scaled_data, target)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_data, target, test_size=0.1, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2],
                     'C': [0.1, 1, 10]}]

print("# Tuning hyper-parameters:")

clf = grid_search.GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
# More reports
TN = FN = FP = TP = 0
for i in range(len(y_pred)):
    if y_true[i] == '0' and '0' == y_pred[i]:
        TN += 1
    if y_true[i] == '1' and '0' == y_pred[i]:
        FN += 1
    if y_true[i] == '0' and '1' == y_pred[i]:
        FP += 1
    if y_true[i] == '1' and '1' == y_pred[i]:
        TP += 1
print "\nTest confusion matrix:"
print "                        ", "Survived", "Died"
print "Predicted as survived   ", "   " + str(TN) + "    ", " " + str(FN) + " "
print "Predicted as Died       ", "   " + str(FP) + "    ", " " + str(TP) + " "
print "Sensitivity: " + str(float(TP) / (TP + FN))
print "Specificity: " + str(float(TN) / (TN + FP))

y_true, y_pred = y_train, clf.predict(X_train)
# More reports
TN = FN = FP = TP = 0
for i in range(len(y_pred)):
    if y_true[i] == '0' and '0' == y_pred[i]:
        TN += 1
    if y_true[i] == '1' and '0' == y_pred[i]:
        FN += 1
    if y_true[i] == '0' and '1' == y_pred[i]:
        FP += 1
    if y_true[i] == '1' and '1' == y_pred[i]:
        TP += 1
print "\nTrain confusion matrix:"
print "                        ", "Survived", "Died"
print "Predicted as survived   ", "   " + str(TN) + "    ", " " + str(FN) + " "
print "Predicted as Died       ", "   " + str(FP) + "    ", " " + str(TP) + " "
print "Sensitivity: " + str(float(TP) / (TP + FN))
print "Specificity: " + str(float(TN) / (TN + FP))
