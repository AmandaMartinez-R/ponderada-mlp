import numpy as np


def relu(z):
    """Aplica a função ReLU elemento a elemento

    Parâmetros:
        z: matriz ou vetor com valores lineares de uma camada.

    Retorno:
        Matriz ou vetor com o mesmo formato de z, mantendo valores positivos
        e substituindo valores negativos por zero.
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """Calcula a derivada da ReLU elemento a elemento.

    Parâmetros:
        z: matriz ou vetor com valores lineares antes da ativacao.

    Retorno:
        Matriz ou vetor com 1 onde z > 0 e 0 onde z <= 0.
    """
    # A ReLU cresce com inclinacao 1 para valores positivos e fica zerada
    # para valores negativos. Em z = 0 usamos 0 por convencao.
    return (z > 0).astype(float)
