import numpy as np
from numpy import random


# normalización de un vector
def normVec(X):
    suma = 0
    for i in range(len(X)):
        suma = suma + (X[i] ** 2)

    suma = suma ** 0.5

    # normalización del vector
    for i in range(len(X)):
        X[i] = X[i] / suma

    return X


# normalización de matrices
def normMat(X):
    for i in range(len(X)):
        normVec(X[i])
    return X


# producto punto del peso y dato de interés
def f(z, w):
    if np.dot(z, w) >= 0:
        return 1
    elif np.dot(z, w) < 0:
        return -1


# consenso de los perceptrones
def s(p):
    if p >= 0:
        return 1
    elif p < 0:
        return -1


# generar la cantidad de pesos especificada con valores aleatorios
def genPesosMat(X, n):
    W = np.zeros([n, len(X[0])])
    W = random.rand(len(W), len(W[0]))
    return W


# extender los vectores de los datos y agregar un uno en la casilla extra
def vec_ext(X):
    X_ext = np.ones([len(X), len(X[0]) + 1])

    for i in range(len(X)):
        for j in range(len(X[0])):
            X_ext[i][j] = X[i][j]

    return X_ext


# regla de aprendizaje delta con los hiperparámetros que nos da el usuario
def PDelta(W, sp, z, label, eps, miu, gamma, t):  # W matriz de pesos, s(p), z entrada de X, label de X
    eta = 1 / (4 * np.sqrt(t))

    for i in range(len(W)):
        if (sp > (label + eps)) and (np.dot(W[i], z) >= 0):
            W[i] = W[i] + (eta * (-1) * z)
        elif (sp < (label - eps)) and (np.dot(W[i], z) < 0):
            W[i] = W[i] + (eta * z)
        elif (sp <= (label + eps)) and (0 <= (np.dot(W[i], z)) and (np.dot(W[i], z) < gamma)):
            W[i] = W[i] + (eta * miu * z)
        elif (sp >= (label - eps)) and ((-1 * gamma) < (np.dot(W[i], z)) and (np.dot(W[i], z) < 0)):
            W[i] = W[i] + (eta * miu * (-1) * z)
        else:
            W[i] = W[i]

    return normMat(W)


# entrenar el modelo con los datos del conjunto de entrenamiento y regresar la matriz de pesos, así como el
# porcentaje de aciertos de las predicciones de la última era
def train(X, Y, n_percep, eps, miu, gamma):
    X = vec_ext(X)  # extender matriz de datos
    W = genPesosMat(X, n_percep)  # generar pesos con base en el tamaño de los datos

    X = normMat(X)  # normalizar la matriz de datos así como la de pesos
    W = normMat(W)

    p = 0
    total = aciertos = 0

    for t in range(1, 10):  # épocas
        total = aciertos = 0
        for i in range(len(X)):
            for j in range(len(W)):
                p = p + f(X[i], W[j])

            if s(p) == Y[i]:
                aciertos += 1
            total += 1

            W = PDelta(W, s(p), X[i], Y[i], eps, miu, gamma, t)
            p = 0

    return W, aciertos / total


# testear los datos del conjunto de prueba y regresar un vector con las predicciones, así como el porcentaje de aciertos
def test(X, Y, W):
    X = vec_ext(X)
    Y_pr = np.zeros(len(X))

    p = 0
    total = aciertos = 0
    for i in range(len(X)):
        for j in range(len(W)):
            p = p + f(X[i], W[j])
        Y_pr[i] = s(p)
        if Y_pr[i] == Y[i]:
            aciertos += 1
        total += 1
        p = 0

    return Y_pr, aciertos / total


# filtrar datos de las clases de interés
def getClasses(X, Y, c1, c2):
    new_X = np.zeros([sum((y == c1 or y == c2) for y in Y), len(X[0])])
    new_Y = np.zeros(sum((y == c1 or y == c2) for y in Y))

    j = 0

    for i in range(len(Y)):
        if Y[i] == c1:
            new_X[j] = X[i]
            new_Y[j] = 1
            j += 1
        if Y[i] == c2:
            new_X[j] = X[i]
            new_Y[j] = -1
            j += 1

    return new_X, new_Y
