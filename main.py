import Perceptron_Paralelo as PP
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# cargar datos
X, Y = load_digits(return_X_y=True)

# establecer hiperparámetros
eps = 0.1  # 0.01
miu = 1
gamma = 0.05
n_percep = 50
porc_tst = 0.2

# establecer clases a identificar
c1 = 3  # 1
c2 = 8  # -1

# filtrar datos a aquellos que coinciden con las clases de interés
X, Y = PP.getClasses(X, Y, c1, c2)

# separar en conjuntos de entrenamiento y de prueba
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=porc_tst, stratify=Y)

# entrenamiento
W, AccTr = PP.train(X_tr, Y_tr, n_percep, eps, miu, gamma)

# prueba
Y_pp, AccTe = PP.test(X_te, Y_te, W)

# imprimir los resultados de ambas fases
print(f'AccTr={AccTr}, AccTe={AccTe}')

# elegir 5 datos de los que se han hecho predicciones
index = np.random.randint(0, len(Y_pp), 5)

# imprimir resultados
for i in index:
    plt.gray()
    plt.matshow(np.reshape(X_te[i], (8, 8)))
    plt.title(Y_pp[i])
    plt.show()
