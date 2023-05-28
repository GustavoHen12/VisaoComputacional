import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

def distance(x_train, x_test_point):
  distances= []
  for row in range(len(x_train)):
      current_train_point= x_train[row]
      current_distance= 0
      for col in range(len(current_train_point)):
          current_distance += (current_train_point[col] - x_test_point[col]) **2
      current_distance= np.sqrt(current_distance)
      distances.append(current_distance)

  distances= pd.DataFrame(data=distances,columns=['dist'])
  return distances

def nearest_neighbors(distance_point, K):
    df_nearest= distance_point.sort_values(by=['dist'], axis=0)
    df_nearest= df_nearest[:K]
    return df_nearest

def voting(df_nearest, y_train):
    counter_vote= Counter(y_train[df_nearest.index])
    y_pred= counter_vote.most_common()[0][0]
    return y_pred

def KNN_from_scratch(x_train, y_train, x_test, K):
    y_pred=[]

    for x_test_point in x_test:
      distance_point  = distance(x_train, x_test_point)
      df_nearest_point= nearest_neighbors(distance_point, K)
      y_pred_point    = voting(df_nearest_point, y_train)
      y_pred.append(y_pred_point)

    return y_pred  

# Carregar o dataset em um DataFrame
iris_df = pd.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Separa o dataset
x= iris_df.iloc[:, :-1]
y= iris_df.iloc[:, -1]

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, shuffle= True, random_state= 0)
x_train= np.asarray(x_train)
y_train= np.asarray(y_train)
x_test= np.asarray(x_test)
y_test= np.asarray(y_test)

# Normaliza o dataset
scaler= Normalizer().fit(x_train)
normalized_x_train= scaler.transform(x_train)
normalized_x_test= scaler.transform(x_test)

K=3
y_pred_scratch= KNN_from_scratch(normalized_x_train, y_train, normalized_x_test, K)
print(y_pred_scratch)

score = accuracy_score(y_test, y_pred_scratch)
print(score)