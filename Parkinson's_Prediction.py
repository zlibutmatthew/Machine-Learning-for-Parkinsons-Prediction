# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/zlibutmatthew/Machine-Learning-for-Parkinsons-Prediction/blob/main/Parkinson's_Prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="5gGSfCjcI4dP"
# ## Parkinson's Disease Data Prediction
# ####      Done By: Sheanel Gardner & Matthew Zlibut

# + [markdown] id="Zm7eJk8HJQxi"
# ## Importing data

# + id="h8ga3XN5IwMl"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix

# + colab={"base_uri": "https://localhost:8080/"} id="Wh6b5oMkI-9e" outputId="9a211212-ace3-4b5d-c7b2-4198c119c5b5"
df = pd.read_csv('https://archive.ics.uci.edu'
                 '/ml/machine-learning-databases/parkinsons'
                 '/parkinsons.data', header=0, 
                 delimiter=r",\s*" ,na_values='?', engine='python')

df.info()

# + colab={"base_uri": "https://localhost:8080/", "height": 224} id="ceBkvPQLJCAU" outputId="70c4972c-81e3-4b9d-fbaf-79db565d717e"
df.head()

# + [markdown] id="8Lb3BPlnJLgI"
# ### Check for missing data

# + colab={"base_uri": "https://localhost:8080/"} id="0KXUXAjgJGev" outputId="1f0b8571-ef82-4df9-cbdb-c3d4928f98ca"
df.isna().sum()

# + [markdown] id="igZXnOa3JdI6"
# ## Preprocessing the data

# + id="A0gCK1-yJXv1"
X = df.drop(['name', 'status'], axis =1) # independent variables
y = df['status'] 

# + colab={"base_uri": "https://localhost:8080/"} id="ogYyS_uTJjZx" outputId="0cc38972-7726-4993-d181-e2a875012cb4"
x= X.values
y = y.values
print(x)
print(y)

# + [markdown] id="lzb6uFRIJoqU"
# ### Find amount of people that have Parkinson's or are healthy

# + colab={"base_uri": "https://localhost:8080/"} id="EC5a_H-UJlDO" outputId="02f6a85f-4723-4adb-e905-ea39369e2381"
parkinson=0
healthy=0
for item in y:
    if item == 0:
        healthy+=1
    else:
        parkinson+=1
        
print("Parkinson's:",parkinson)
print('Healthy:',healthy)

# + [markdown] id="e7nFZVOAKFEY"
# ### Split into training and testing data sets

# + id="b-JJhqGyJ35I"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, 
                     test_size=0.25,
                     stratify=y,
                     random_state=1)

# + colab={"base_uri": "https://localhost:8080/"} id="6zDS0utpKNWa" outputId="7961d6d0-aa45-4306-8dcf-a562ad109979"
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# + [markdown] id="rnDmN5KVKWvM"
# ### Normalize the Data

# + colab={"base_uri": "https://localhost:8080/"} id="ZJ-HLsD1KRAm" outputId="21d9a59a-668b-47e2-d501-b8f4c5ad7476"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
print(X_train_std[:1,:])
print(X_test_std[:1, :])

# + [markdown] id="BNnX2GqtKgv3"
# ## Logistic Regression

# + colab={"base_uri": "https://localhost:8080/"} id="B8d7l10_Kcdh" outputId="5f9674e6-9b57-43e9-9943-856fc6492cf7"
from sklearn.linear_model import LogisticRegression

param_grid = [{'C': np.logspace(-4, 2, 7)}]
lr = LogisticRegression()
gs = GridSearchCV(estimator=lr, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=10,
    n_jobs=-1)
gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print('Accuracy Training: ', gs.best_score_)
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
print('Accuracy Test: ', accuracy_score(y_test, y_pred_test))

print('Precision: ', precision_score(y_train, y_pred_train))
print('Recall: ', recall_score(y_train, y_pred_train))
print('F1: ', f1_score(y_train, y_pred_train))

print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="qDGB2rqiKxKV" outputId="177610a7-d636-4241-d70a-37dc4a280277"
array_test=confusion_matrix(y_test, y_pred_test)
df_cm=pd.DataFrame(array_test)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

# + [markdown] id="68zQwJuvK-3P"
# ## Support Vector Machine

# + id="M7LikCXzLF4l"
C_range = np.logspace(-4, 2, 7)
gamma_range = np.logspace(-4, 2, 7)
param_grid = [{'C': C_range, 'kernel': ['linear']},
    {'C': C_range, 
    'gamma': gamma_range, 
    'kernel': ['rbf']}]

# + colab={"base_uri": "https://localhost:8080/"} id="m6g6rYO3LLcL" outputId="47f17c0e-1247-44e7-b20c-183ee1c8839e"
from sklearn.svm import SVC

svc = SVC()
gs = GridSearchCV(estimator=svc, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=10,
    n_jobs=-1)
gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print('Accuracy Train: ', gs.best_score_)
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
print('Accuracy Test: ', accuracy_score(y_test, y_pred_test))
print('Precision: ', precision_score(y_train, y_pred_train))
print('Recall: ', recall_score(y_train, y_pred_train))
print('F1: ', f1_score(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="F8WoiBNZLT9k" outputId="1454f10a-27de-459b-bda6-b69efe59517e"
array_test=confusion_matrix(y_test, y_pred_test)
df_cm=pd.DataFrame(array_test)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

# + [markdown] id="baQI8B8ULljC"
# ## Multi-Layer Perceptron Classifier

# + colab={"base_uri": "https://localhost:8080/"} id="kO-yoQm5LtCS" outputId="c96be6ae-f304-4b4c-ad87-5f82aa5496a0"
hls_range = [(8,8), (10,8), (12,8), (20,10), (50,8), (100,50)]
alpha_range = np.logspace(-2,2,5)

param_grid = [{'alpha':alpha_range, 'hidden_layer_sizes':hls_range}]


gs = GridSearchCV(estimator=MLPClassifier(tol=1e-5, 
                                          learning_rate_init=0.02,
                                          max_iter=1000,
                                         random_state=1), 
                  param_grid=param_grid, 
                  cv=5)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)

y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)
print("The accuracy for the training data is :", gs.best_estimator_.score(X_train_std,y_train))
print("The accuracy for the test data is :",gs.best_estimator_.score(X_test_std,y_test))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="A2g5AhlAL28m" outputId="d909c2f5-77ad-4b0c-883c-8210d6999ee9"
array_test=confusion_matrix(y_test, y_pred_test)
df_cm=pd.DataFrame(array_test)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
