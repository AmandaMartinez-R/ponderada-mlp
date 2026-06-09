import numpy as np

from mlp.activations import relu
from mlp.losses import softmax


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
