import numpy as np
import pandas as pd

df = pd.read_csv('heart.csv')

X = df.iloc[: , :-1]
Y = df.target

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(X, Y)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)

from sklearn import metrics

# predicted = clf.predict(x_test)
# print(metrics.classification_report(y_test, predicted))

# clf = KNeighborsClassifier(n_neighbors=7)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print(metrics.classification_report(y_test, y_pred))

from sklearn.svm import SVC

# clf = SVC(C=15, kernel='linear')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print(metrics.classification_report(y_test, y_pred))

# Gridsearch

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

params = {
    'C': range(1, 10, 0.1),
    'kernel': ['linear', 'poly'],
    'gamma': ['auto', 0.1]
}
gsh = GridSearchCV(SVC(), param_grid=params, scoring='accuracy', cv=5, n_jobs=-1, verbose=10)
gsh.fit(X, Y)

print(gsh.best_estimator_)
print(gsh.best_score_)
print(gsh.best_params_)