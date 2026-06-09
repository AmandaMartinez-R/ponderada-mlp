import numpy as np

from mlp.activations import relu, relu_derivative
from mlp.losses import softmax
from mlp.optimizers import sgd_step


class MLP:
    """Rede neural MLP implementada do zero com NumPy."""

    def __init__(self, layer_sizes, learning_rate=0.01, seed=None):
        """Inicializa a arquitetura da rede e seus parâmetros.

        Parâmetros:
            layer_sizes: lista com o tamanho de cada camada.
            learning_rate: taxa de aprendizado usada no treinamento.
            seed: semente opcional para reproduzir os resultados.
        """
        if len(layer_sizes) < 2:
            raise ValueError(
                "A rede precisa ter pelo menos uma camada de entrada e uma camada de saída."
            )

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        self.cache = {}
        self.gradients = {}

        self.initialize_parameters()

    def initialize_parameters(self):
        """Inicializa pesos com He Initialization e vieses com zero."""
        self.weights = []
        self.biases = []

        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # He Initialization é adequada para ReLU porque usa a quantidade
            # de entradas da camada para manter a escala dos sinais mais estável.
            weight = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(2 / input_size),
                size=(input_size, output_size),
            )
            bias = np.zeros((1, output_size))

            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        """Executa o forward pass da rede.

        Parâmetros:
            X: matriz de entrada com shape (batch_size, input_size).

        Retorno:
            Probabilidades da camada de saída com shape (batch_size, output_size).
        """
        activation = X
        self.cache = {"A": [X], "Z": []}

        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Cada camada primeiro aplica a transformação linear Z = A anterior @ W + b.
            z = activation @ weight + bias
            self.cache["Z"].append(z)

            is_output_layer = layer_index == len(self.weights) - 1
            if is_output_layer:
                # Na saída, o softmax transforma logits em probabilidades por classe.
                activation = softmax(z)
            else:
                # Nas camadas ocultas, a ReLU adiciona não linearidade ao modelo.
                activation = relu(z)

            self.cache["A"].append(activation)

        return activation

    def backward(self, y_true):
        """Executa o backpropagation e calcula os gradientes da rede.

        Parâmetros:
            y_true: rótulos verdadeiros em one-hot com shape
                (batch_size, output_size).

        Retorno:
            Dicionário com os gradientes `dW` e `db` de cada camada.
        """
        if not self.cache or "A" not in self.cache or "Z" not in self.cache:
            raise ValueError("Execute o forward pass antes do backward pass.")

        y_pred = self.cache["A"][-1]
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true deve ter o mesmo shape da saída da rede.")

        batch_size = y_true.shape[0]
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Para softmax + cross-entropy, o gradiente da saída simplifica para
        # a diferença entre a probabilidade prevista e o rótulo verdadeiro.
        dZ = y_pred - y_true

        for layer_index in reversed(range(len(self.weights))):
            previous_activation = self.cache["A"][layer_index]

            dW[layer_index] = previous_activation.T @ dZ / batch_size
            db[layer_index] = np.sum(dZ, axis=0, keepdims=True) / batch_size

            if layer_index > 0:
                dA_previous = dZ @ self.weights[layer_index].T
                previous_z = self.cache["Z"][layer_index - 1]
                dZ = dA_previous * relu_derivative(previous_z)

        self.gradients = {"dW": dW, "db": db}
        return self.gradients

    def update_parameters(self):
        """Atualiza pesos e vieses usando SGD."""
        if not self.gradients:
            raise ValueError("Execute o backward pass antes de atualizar os parâmetros.")

        sgd_step(
            weights=self.weights,
            biases=self.biases,
            gradients=self.gradients,
            learning_rate=self.learning_rate,
        )
