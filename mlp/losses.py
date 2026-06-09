import numpy as np


def one_hot_encode(y, num_classes):
    """Converte rotulos inteiros para representacao one-hot.

    Parametros:
        y: vetor com rotulos inteiros, como [0, 3, 9] v=       num_classes: quantidade total de classes.

    Retorno:
        Matriz com shape (quantidade_de_amostras, num_classes).
    """
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def softmax(logits):
    """Transforma logits em probabilidades por classe.

    Parametros:
        logits: matriz com shape (batch_size, num_classes).

    Retorno:
        Matriz de probabilidades com o mesmo shape dos logits.
    """
    # Subtrair o maior valor de cada linha evita overflow no np.exp.
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted_logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
    """Calcula a cross-entropy media de um batch.

    Parametros:
        y_pred: probabilidades previstas com shape (batch_size, num_classes).
        y_true: rotulos em one-hot com shape (batch_size, num_classes).
        epsilon: valor pequeno para evitar log(0).

    Retorno:
        Valor escalar com a loss media do batch.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    losses = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(losses)
