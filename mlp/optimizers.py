def sgd_step(weights, biases, gradients, learning_rate):
    """Atualiza pesos e vieses usando Stochastic Gradient Descent.

    Parâmetros:
        weights: lista de matrizes de pesos da rede.
        biases: lista de vetores de vieses da rede.
        gradients: dicionário com listas `dW` e `db`.
        learning_rate: tamanho do passo dado na direção oposta ao gradiente.

    Retorno:
        Nenhum. A atualização é feita diretamente nas listas recebidas.
    """
    if "dW" not in gradients or "db" not in gradients:
        raise ValueError("Os gradientes devem conter as chaves 'dW' e 'db'.")

    if len(weights) != len(gradients["dW"]) or len(biases) != len(gradients["db"]):
        raise ValueError("A quantidade de gradientes deve bater com a quantidade de parâmetros.")

    for layer_index in range(len(weights)):
        # O SGD anda na direção oposta ao gradiente para reduzir a loss.
        weights[layer_index] -= learning_rate * gradients["dW"][layer_index]
        biases[layer_index] -= learning_rate * gradients["db"][layer_index]
