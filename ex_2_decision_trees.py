# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

# %%

df = pd.read_csv('stud_cyto2.csv', index_col=0)

print("Zad. 1:")
print(df.iloc[15:21, 0])

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :3], df.iloc[:, 3], test_size=0.25, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

feature_names = df.iloc[:, :3].columns
class_names = ['non-cyt', 'cyt']

A = [[0, -0.24, -2.5]]
B = [[0, -0.17, -1]]

print("\nZad. 2:")
print(f'Punkt A: {class_names[clf.predict(A)[0]]}')
print(f'Punkt B: {class_names[clf.predict(B)[0]]}')

# i_list = []
# test_list = []
# train_list = []
# for i in range(1, 20):
#     clf = DecisionTreeClassifier(max_depth=i, random_state=42)
#     clf = clf.fit(X_train, y_train)
#     test_score = clf.score(X_test, y_test)
#     train_score = clf.score(X_train, y_train)
#     i_list.append(i)
#     test_list.append(test_score)
#     train_list.append(train_score)
# plt.plot(i_list, train_list)
# plt.plot(i_list, test_list)

clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, class_names=class_names, feature_names=feature_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# %%

df = pd.read_csv('stud_cyto3.csv', index_col=0)

print("Zad. 1:")
print(df.iloc[15:21, 0])

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :4], df.iloc[:, 4], test_size=0.25, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

feature_names = df.iloc[:, :4].columns
class_names = ['non-cyt', 'cyt']

A = [[0, 0, -0.24, -2.5]]
B = [[0, 0, -0.17, -1]]

print("\nZad. 2:")
print(f'Punkt A: {class_names[clf.predict(A)[0]]}')
print(f'Punkt B: {class_names[clf.predict(B)[0]]}')

i_list = []
test_list = []
train_list = []
for i in range(1, 20):
    clf = DecisionTreeClassifier(max_depth=i, random_state=42)
    clf = clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    train_score = clf.score(X_train, y_train)
    i_list.append(i)
    test_list.append(test_score)
    train_list.append(train_score)
plt.plot(i_list, train_list)
plt.plot(i_list, test_list)
plt.show()

# clf = DecisionTreeClassifier(max_depth=4)
# clf = clf.fit(X_train, y_train)

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
#                 special_characters=True, class_names=class_names, feature_names=feature_names)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
# %%
