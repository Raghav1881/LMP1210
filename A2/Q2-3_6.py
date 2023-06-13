# %%
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# %%
def load_data1():
  data_dir = '/content/hw1_data.csv'
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

  X_train = train_data.drop('Dataset', axis = 1)
  y_train = train_data['Dataset']
  X_val = val_data.drop('Dataset', axis = 1)
  y_val = val_data['Dataset']
  X_test = test_data.drop('Dataset', axis = 1)
  y_test = test_data['Dataset']
  
  return X_train, y_train, X_val, y_val, X_test, y_test

# %% [markdown]
# # **QUESTION 2** a)

# %%
from sklearn.ensemble import RandomForestClassifier

X_train, y_train, X_val, y_val, X_test, y_test = load_data1()
n_estimators_values = [10, 20, 50, 100]
train_accuracies = []
val_accuracies = []

for n_estimators in n_estimators_values:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

plt.plot(n_estimators_values, train_accuracies, label='Training accuracy')
plt.plot(n_estimators_values, val_accuracies, label='Validation accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

best_n_estimators = n_estimators_values[np.argmax(val_accuracies)]
clf = RandomForestClassifier(n_estimators=best_n_estimators)
clf.fit(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f'Test accuracy for the best Random Forest model with n_estimators = {best_n_estimators}: test_acc = {test_acc * 100:.2f}%')

# %% [markdown]
# The random forest classifier had a better accuracy of 72.65% in comparison to the logistic regression model (70.09%)

# %% [markdown]
# # **2 b)**

# %%
from sklearn.inspection import permutation_importance

X_train, y_train, X_val, y_val, X_test, y_test = load_data1()

# Using best n estimators value of 10 from a)
clf = RandomForestClassifier(n_estimators=best_n_estimators)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

plt.barh(range(X_train.shape[1]), importances[sorted_idx])
plt.xlabel('Relative feature importance')
plt.title('Feature importance using clf.feature_importances_')
plt.show()

result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=0, scoring='accuracy')
perm_importance = result.importances_mean
sorted_idx = np.argsort(perm_importance)

plt.barh(range(X_train.shape[1]), perm_importance[sorted_idx])
plt.xlabel('Relative feature importance')
plt.title('Feature importance using permutation_importance')
plt.show()

# %% [markdown]
# # **QUESTION 3 a)**

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

X, y = make_classification(n_samples=1000, n_features=20, weights=[0.90,0.10])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot the ROC and PR curves
y_probs = model.predict_proba(X_val)
precision, recall, thresholds = precision_recall_curve(y_val, y_probs[:,1])
fpr, tpr, thresholds = roc_curve(y_val, y_probs[:,1])

plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend(loc='best')
plt.show()

# %% [markdown]
# 3 B)

# %%
X, y = make_classification(n_samples=1000, n_features=20, weights=[0.90,0.10])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
model = LogisticRegression(class_weight = 'balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot the ROC and PR curves
y_probs = model.predict_proba(X_val)
precision, recall, thresholds = precision_recall_curve(y_val, y_probs[:,1])
fpr, tpr, thresholds = roc_curve(y_val, y_probs[:,1])

plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend(loc='best')
plt.show()

# %% [markdown]
# I was able to improve recall significantly while also gaining precision

# %% [markdown]
# # **Question 6 a)**

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE

df = pd.read_csv('/content/hw2_data.csv')
features = df.loc[:, df.columns != 'Cell Type']
values = df['Cell Type']
features.index = values.astype('category')
print(features.index)
enc = LabelEncoder()
values = enc.fit_transform(values.astype('str'))
X_train, X_val, y_train, y_val = train_test_split(features, values, test_size=0.3)

xgb = XGBClassifier(m_estimators = 100, learning_rate = 0.1, max_depth = 3)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)

xgb_acc = accuracy_score(y_val, xgb_pred)
print("Accuracy of the XGBoost model:", xgb_acc)

mlp = MLPClassifier(early_stopping = True)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_val)

accuracy_mlp = accuracy_score(y_val, y_pred_mlp)
print("Accuracy of the Multi-layer Perceptron model:", accuracy_mlp)

# %%
x = features.to_numpy()
x

# %% [markdown]
# # **6 b)**

# %%
import seaborn as sns
from xgboost import plot_importance
from matplotlib import pyplot

xgb_importance = xgb.feature_importances_
top30_genes_xgb = features.columns[xgb_importance.argsort()[::-1][:30]]

mlp_importance = np.abs(mlp.coefs_[0]).mean(axis=0)
top30_genes_mlp = features.columns[mlp_importance.argsort()[::-1][:30]]

data_top30_genes = df.loc[:, list(top30_genes_xgb) + list(top30_genes_mlp) + ['Cell Type']]
data_top30_genes_avg = data_top30_genes.groupby('Cell Type').mean()
sns.set(font_scale=1.2)
fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(data_top30_genes_avg.T, cmap='coolwarm', ax=ax)
ax.set_xlabel('Cell type')
plt.show()

# %% [markdown]
# My interpretation of this heatmap is that these listed genes are most likely candidate marker genes to delineate the identity of individual cell types based on their expression of a specific gene. These marker genes could be used to verify whether a cell is a B cell, or a CD14 monocyte, with the first 12 genes showing clear opposing levels of expression, suggesting that these genes are highly expressed specifically to one cell type


