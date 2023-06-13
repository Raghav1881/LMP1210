# %%
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# %% [markdown]
# QUESTION 4 a)

# %%
def load_data():
  data_dir = '/content/data/hw1_data.csv'
  df = pd.read_csv(data_dir)
  df = df.fillna(0)
  df = df.sample(frac = 1)
  enc = LabelEncoder()
  df['Gender'] = enc.fit_transform(df['Gender'].astype('str'))
  split_1 = int(0.7 * len(df))
  split_2 = int(0.8 * len(df))
  train_data = df.iloc[:split_1]
  val_data = df.iloc[split_1:split_2]
  test_data = df.iloc[split_2:]
  return train_data, val_data, test_data

# %%
train_data, val_data, test_data = load_data()

# %%
train_data

# %% [markdown]
# QUESTION 4 b)

# %%
def select_knn_model(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']
  train_accuracies = []
  val_accuracies = []
  k_values = range(1, 21)

  for k in k_values:
      knn = KNeighborsClassifier(n_neighbors=k)
      knn.fit(X_train, y_train)
      train_acc = accuracy_score(y_train, knn.predict(X_train))
      val_acc = accuracy_score(y_val, knn.predict(X_val))
      train_accuracies.append(train_acc)
      val_accuracies.append(val_acc)

  plt.plot(k_values, train_accuracies, label='Training Accuracy')
  plt.plot(k_values, val_accuracies, label='Validation Accuracy')
  plt.xlabel('k')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()
  best_k = k_values[val_accuracies.index(max(val_accuracies))]
  best_knn = KNeighborsClassifier(n_neighbors=best_k)
  best_knn.fit(X_train,y_train)
  test_acc = accuracy_score(y_test, best_knn.predict(X_test))
  print(f"Test accuracy: {test_acc}")
  return best_knn, test_acc

# %%
select_knn_model(train_data, val_data, test_data)

# %% [markdown]
# QUESTION 4 c)

# %%
def select_knn_model(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']
  train_accuracies = []
  val_accuracies = []
  k_values = range(1, 21)

  for k in k_values:
      knn = KNeighborsClassifier(n_neighbors=k, metric = 'cosine')
      knn.fit(X_train, y_train)
      train_acc = accuracy_score(y_train, knn.predict(X_train))
      val_acc = accuracy_score(y_val, knn.predict(X_val))
      train_accuracies.append(train_acc)
      val_accuracies.append(val_acc)

  plt.plot(k_values, train_accuracies, label='Training Accuracy')
  plt.plot(k_values, val_accuracies, label='Validation Accuracy')
  plt.xlabel('k')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()
  best_k = k_values[val_accuracies.index(max(val_accuracies))]
  best_knn = KNeighborsClassifier(n_neighbors=best_k)
  best_knn.fit(X_train,y_train)
  test_acc = accuracy_score(y_test, best_knn.predict(X_test))
  print(f"Test accuracy: {test_acc}")
  return best_knn, test_acc

# %%
select_knn_model(train_data, val_data, test_data)

# %% [markdown]
# QUESTION 5 a)

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
def train_decision_tree(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']

  clf = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_leaf=1)
  clf.fit(X_train, y_train)
  train_acc = clf.score(X_train, y_train)
  print("Training accuracy: {:.2f}%".format(train_acc * 100))
  
  val_acc = clf.score(X_val, y_val)
  print("Validation accuracy: {:.2f}%".format(val_acc * 100))
  
  test_acc = clf.score(X_test, y_test)
  print("Test accuracy: {:.2f}%".format(test_acc * 100))

# %%
train_decision_tree(train_data, val_data, test_data)

# %%
# min_sample_leaf=2
def train_decision_tree(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']

  clf = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_leaf=2)
  clf.fit(X_train, y_train)
  train_acc = clf.score(X_train, y_train)
  print("Training accuracy: {:.2f}%".format(train_acc * 100))
  
  val_acc = clf.score(X_val, y_val)
  print("Validation accuracy: {:.2f}%".format(val_acc * 100))
  
  test_acc = clf.score(X_test, y_test)
  print("Test accuracy: {:.2f}%".format(test_acc * 100))

# %%
train_decision_tree(train_data, val_data, test_data)

# %%
# min_sample_leaf=3
def train_decision_tree(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']

  clf = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_leaf=2)
  clf.fit(X_train, y_train)
  train_acc = clf.score(X_train, y_train)
  print("Training accuracy: {:.2f}%".format(train_acc * 100))
  
  val_acc = clf.score(X_val, y_val)
  print("Validation accuracy: {:.2f}%".format(val_acc * 100))
  
  test_acc = clf.score(X_test, y_test)
  print("Test accuracy: {:.2f}%".format(test_acc * 100))

# %%
train_decision_tree(train_data, val_data, test_data)

# %% [markdown]
# QUESTION 5 b)

# %%
# min_sample_leaf=2
def train_decision_tree(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']

  clf = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_leaf=2)
  clf.fit(X_train, y_train)
  train_acc = clf.score(X_train, y_train)
  print("Training accuracy: {:.2f}%".format(train_acc * 100))
  
  val_acc = clf.score(X_val, y_val)
  print("Validation accuracy: {:.2f}%".format(val_acc * 100))
  
  test_acc = clf.score(X_test, y_test)
  print("Test accuracy: {:.2f}%".format(test_acc * 100))
  return clf

# %%
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source

clf = train_decision_tree(train_data, val_data, test_data)
graph = Source(export_graphviz(clf, out_file=None, feature_names=['Age','Total_Bilirubin','Direct_Bilirubin',
                                                  'Alkaline_Phosphotase','Alamine_Aminotransferase',
                                                  'Aspartate_Aminotransferase',
                                                  'Total_Protiens', 'Albumin',
                                                  'Albumin_and_Globulin_Rati', 'Dataset']))
graph.format = 'png'
graph.render('dt', view=True)

# %% [markdown]
# QUESTION 6

# %%
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def train_logreg_model(train_data, val_data, test_data):
  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']

  logistic_regression = LogisticRegression(random_state=0, max_iter = 10000).fit(X_train, y_train)
  logistic_regression.fit(X_train, y_train)

  train_acc = accuracy_score(y_train, logistic_regression.predict(X_train))
  val_acc = accuracy_score(y_val, logistic_regression.predict(X_val))
  test_acc = accuracy_score(y_test, logistic_regression.predict(X_test))
  
  print("Training accuracy: {:.2f}%".format(train_acc * 100))
  print("Validation accuracy: {:.2f}%".format(val_acc * 100))
  print("Test accuracy: {:.2f}%".format(test_acc * 100))

# %%
 train_logreg_model(train_data, val_data, test_data)


