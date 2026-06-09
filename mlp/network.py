import numpy as np

from mlp.activations import relu, relu_derivative
from mlp.losses import cross_entropy_loss, one_hot_encode, softmax
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
        self.history = {}

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

    def train_batch(self, X_batch, y_batch):
        """Treina a rede em um mini-batch.

        Parâmetros:
            X_batch: matriz de entradas com shape (batch_size, input_size).
            y_batch: rótulos inteiros com shape (batch_size,).

        Retorno:
            Loss média do batch após o forward pass.
        """
        y_true = one_hot_encode(y_batch, self.layer_sizes[-1])
        y_pred = self.forward(X_batch)
        loss = cross_entropy_loss(y_pred, y_true)

        self.backward(y_true)
        self.update_parameters()

        return loss

    def fit(self, X_train, y_train, epochs=10, batch_size=64, shuffle=True):
        """Treina a rede usando mini-batches.

        Parâmetros:
            X_train: matriz de treino com shape (num_amostras, input_size).
            y_train: vetor de rótulos inteiros com shape (num_amostras,).
            epochs: quantidade de passagens completas pelo conjunto de treino.
            batch_size: quantidade de amostras usadas em cada atualização.
            shuffle: se True, embaralha os dados no início de cada época.

        Retorno:
            Dicionário com o histórico de loss média por época.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train e y_train devem ter a mesma quantidade de amostras.")
        if X_train.shape[0] == 0:
            raise ValueError("O conjunto de treino não pode estar vazio.")
        if epochs <= 0:
            raise ValueError("epochs deve ser maior que zero.")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero.")

        num_samples = X_train.shape[0]
        self.history = {"loss": []}

        for _ in range(epochs):
            if shuffle:
                indices = self.rng.permutation(num_samples)
                X_epoch = X_train[indices]
                y_epoch = y_train[indices]
            else:
                X_epoch = X_train
                y_epoch = y_train

            batch_losses = []

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]
                batch_loss = self.train_batch(X_batch, y_batch)
                batch_losses.append(batch_loss)

            self.history["loss"].append(float(np.mean(batch_losses)))

        return self.history

    def predict_proba(self, X):
        """Retorna as probabilidades previstas para cada classe."""
        return self.forward(X)

    def predict(self, X):
        """Retorna a classe prevista para cada amostra."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y):
        """Calcula a acurácia da rede para um conjunto de dados."""
        y = np.asarray(y)
        predictions = self.predict(X)

        if predictions.shape[0] != y.shape[0]:
            raise ValueError("X e y devem ter a mesma quantidade de amostras.")

        return float(np.mean(predictions == y))
