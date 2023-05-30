from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
import random
from statistics import mode
from statistics import mean

X, y = load_iris(return_X_y=True)

tam = 1000

seeds = random.sample(range(1, 10000 + 1), tam)
melhores_viz = []

for i in seeds:

  menor_rmse = 10000;
  melhor_nr_vizinhos = 0;

  for j in range(1,21):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=i)

    n_neighbors = j

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    confusion_matrix(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    if (rmse < menor_rmse):
      menor_rmse = rmse
      melhor_nr_vizinhos = j

  print(i)
  print("menor_rmse   : ", np.round(menor_rmse, 2))
  print("melhor nr viz: ", melhor_nr_vizinhos)
  print("----")
  melhores_viz.append(melhor_nr_vizinhos)

print(mode(melhores_viz))
print(mean(melhores_viz))