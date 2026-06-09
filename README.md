# MLP do Zero

## Registro de Desenvolvimento

--- 

## Fases de Planejamento

### Fase 1 - Compreensão da Atividade e Análise do Repositório

---

### Fase 2 - Entendimento Conceitual do MLP

---

### Fase 3 - Planejamento dos Arquivos

--- 

## Etapas de Implementação

### Etapa 1 - Função ReLU e Derivada

Nesta etapa, implementei a função ReLU e sua derivada. Entendi que a ReLU é importante porque adiciona não linearidade à rede, permitindo que o MLP aprenda padrões mais complexos do que uma simples transformação linear.

Também registrei que, no ponto zero, a derivada não é definida matematicamente, então adotei o valor 0 por convenção, que é uma escolha comum em redes neurais.

Teste realizado:

```text
Entrada: valores negativos, zero e positivos
Resultado esperado: negativos viram 0, positivos permanecem iguais
Status: passou
```

### Etapa 2 - Softmax e Cross-Entropy

Nesta etapa, implementei o softmax, a cross-entropy loss e a conversão dos rótulos para one-hot encoding. Entendi que o softmax transforma os valores finais da rede em probabilidades, enquanto a cross-entropy mede o quanto a probabilidade atribuída à classe correta está distante do ideal.

Também aprendi que é importante aplicar estabilidade numérica no softmax, subtraindo o maior logit de cada linha antes da exponenciação. Além disso, usei `clip` na cross-entropy para evitar `log(0)`.

Testes realizados:

```text
Softmax: as probabilidades de cada linha somam 1
One-hot: os rótulos inteiros foram convertidos para vetores corretamente
Cross-entropy: a loss foi calculada para um batch simples
Status: passou
```

### Etapa 3 - Inicialização dos Pesos

Nesta etapa, implementei a inicialização dos pesos da rede na classe `MLP`. Antes de escolher a estratégia, comparei a ideia de inicializar tudo com zero, Xavier Initialization e He Initialization.

Inicializar todos os pesos com zero não seria uma boa escolha, porque todos os neurônios de uma mesma camada começariam iguais e receberiam gradientes muito parecidos. Isso faria com que eles aprendessem praticamente a mesma coisa, impedindo a rede de aproveitar vários neurônios para capturar padrões diferentes.

Também considerei Xavier Initialization, que costuma ser uma boa escolha para ativações como `sigmoid` e `tanh`, porque tenta manter a variância dos sinais equilibrada entre as camadas. Porém, neste projeto a ativação principal das camadas ocultas será a ReLU.

Escolhi He Initialization justamente porque ela foi pensada para redes que usam ReLU. Como a ReLU zera os valores negativos, parte dos sinais deixa de passar para a próxima camada. A He Initialization compensa esse efeito usando uma escala baseada em `sqrt(2 / quantidade_de_entradas_da_camada)`, ajudando a manter os valores propagados em uma faixa mais estável durante o forward pass.

A fórmula usada foi:

```text
W = valores aleatórios com média 0 e desvio padrão sqrt(2 / fan_in)
b = zeros
```

Onde `fan_in` representa a quantidade de entradas da camada. Por exemplo, em uma camada que recebe 784 valores de entrada, os pesos dessa camada são inicializados usando `sqrt(2 / 784)`.

Testes realizados:

```text
Pesos: formatos compatíveis com as camadas
Biases: inicializados com zeros
Reprodutibilidade: mesma seed gera os mesmos pesos
Validação de arquitetura: redes com menos de duas camadas geram erro
Status: passou
```


### Etapa 4 - Forward Pass

Nesta etapa, implementei o forward pass da rede. Entendi que o forward pass é o caminho da entrada até a saída: cada camada recebe ativações da camada anterior, aplica uma transformação linear com pesos e vieses, e depois aplica uma função de ativação.

Nas camadas ocultas usei ReLU, porque ela adiciona não linearidade ao modelo. Na camada final usei softmax, porque o problema é de classificação multiclasse e a saída precisa representar probabilidades para os 10 dígitos do MNIST.

Também passei a armazenar os valores intermediários em `cache`, separando `Z` e `A`. Esse armazenamento será importante no backpropagation, porque os gradientes dependem dos valores calculados durante o forward pass.

Testes realizados:

```text
Forward: saída com shape esperado
Softmax: probabilidades da saída somam 1 por amostra
Cache: ativações e valores lineares foram armazenados
Arquitetura arbitrária: forward funciona com mais de uma camada oculta
Status: passou
```


### Etapa 5 - Backpropagation

Nesta etapa, implementei o backpropagation manual da rede. Entendi que essa é a parte em que a rede calcula quanto cada peso e cada viés contribuíram para o erro final, propagando o erro da saída de volta até as primeiras camadas.

Na camada de saída, usei a simplificação do gradiente de softmax com cross-entropy. Em vez de calcular as derivadas separadamente, o gradiente inicial fica:

```text
dZ = y_pred - y_true
```

Depois disso, para cada camada, calculei os gradientes dos pesos e dos vieses usando as ativações armazenadas no forward pass:

```text
dW = A_anterior.T @ dZ / batch_size
db = soma(dZ) / batch_size
```

Para voltar para as camadas ocultas, propaguei o gradiente pelos pesos da camada atual e multipliquei pela derivada da ReLU. Essa parte foi importante porque a ReLU bloqueia o gradiente onde o valor linear era menor ou igual a zero.

Também percebi que o cache criado no forward pass é essencial para o backward pass. Sem os valores intermediários `A` e `Z`, eu não teria como calcular os gradientes de cada camada corretamente.

Testes realizados:

```text
Backward: gradientes calculados para todas as camadas
Shapes: dW e db têm os mesmos formatos de W e b
Validação: backward exige que o forward seja executado antes
Checagem numérica: gradientes analíticos próximos dos gradientes aproximados
Status: passou
```

### Etapa 6 - SGD

Nesta etapa, implementei o Stochastic Gradient Descent (SGD), que é o algoritmo responsável por atualizar os pesos e vieses da rede usando os gradientes calculados no backpropagation.

Entendi que o backpropagation apenas calcula a direção em que cada parâmetro influencia o erro, mas quem realmente altera os parâmetros é o otimizador. No caso do SGD, a atualização segue a regra:

```text
parâmetro = parâmetro - learning_rate * gradiente
```

A subtração é importante porque queremos andar na direção oposta ao gradiente, ou seja, na direção que tende a reduzir a loss. Se eu somasse o gradiente em vez de subtrair, a tendência seria aumentar o erro.

Também percebi que o `learning_rate` controla o tamanho do passo. Um valor muito alto pode fazer a rede passar do ponto ideal e oscilar; um valor muito baixo pode fazer o treinamento ficar lento demais.

Separei a função `sgd_step` no arquivo `optimizers.py` para manter a lógica de atualização fora da classe principal da rede. Na classe `MLP`, criei o método `update_parameters`, que usa os gradientes armazenados depois do backward pass.

Testes realizados:

```text
SGD: pesos e vieses foram atualizados na direção correta
Learning rate: mudança nos parâmetros bateu com lr * gradiente
Validação: update_parameters exige que o backward seja executado antes
Loss: um passo pequeno de SGD reduziu a loss em um batch simples
Status: passou
```

### Etapa 7 - Mini-Batch Training

Nesta etapa, implementei o treinamento por mini-batches. Entendi que essa estratégia fica no meio-termo entre atualizar os pesos usando uma única amostra por vez e usar o dataset inteiro em uma única atualização.

Com mini-batches, a rede calcula o forward pass, a loss, o backpropagation e o SGD usando pequenos blocos de dados. Isso deixa o treino mais eficiente e costuma tornar as atualizações mais estáveis do que usar apenas uma amostra por vez.

Criei o método `train_batch`, que executa uma etapa completa de treinamento em um batch: converte os rótulos para one-hot, faz o forward pass, calcula a cross-entropy, executa o backpropagation e atualiza os parâmetros com SGD.

Também criei o método `fit`, que organiza o treinamento por épocas. Em cada época, os dados podem ser embaralhados e depois divididos em mini-batches. Tomei cuidado para embaralhar `X` e `y` usando os mesmos índices, porque se eles fossem embaralhados separadamente, as entradas perderiam correspondência com seus rótulos.

Nesta etapa, o histórico de treinamento passou a armazenar a loss média de cada época. Esse histórico será importante depois para gerar a curva de loss nos resultados.

Testes realizados:

```text
train_batch: retorna uma loss escalar
fit: retorna um histórico com uma loss por época
Mini-batches: todos os exemplos são usados, inclusive o último batch menor
Shuffle: X e y mantêm correspondência por usarem os mesmos índices
Loss: treinamento simples reduz a loss ao longo das épocas
Validações: entradas vazias, batch_size inválido e epochs inválido geram erro
Status: passou
```

### Etapa 8 - Treinamento Inicial

Nesta etapa, fiz o primeiro treinamento completo da rede em um problema pequeno e controlado antes de partir para o MNIST. A ideia foi validar se todas as partes implementadas até aqui funcionam juntas: forward pass, cálculo da loss, backpropagation, SGD e treinamento por mini-batches.

Também adicionei métodos de avaliação na classe `MLP`: `predict_proba`, `predict` e `accuracy`. Com eles, ficou possível medir não apenas se a loss diminui, mas também se as previsões da rede melhoram após o treinamento.

Usei um dataset sintético simples com duas classes separáveis. Esse tipo de teste é útil porque, se a rede não conseguisse aprender um problema pequeno, provavelmente haveria algum erro na implementação antes mesmo de testar no MNIST.

Antes do treinamento, a rede ainda fazia previsões ruins, como esperado para pesos aleatórios. Depois de treinar por algumas épocas, a loss caiu bastante e a acurácia chegou a 100% nesse conjunto simples.

Resultados do teste inicial:

```text
Loss antes do treino: 0.879727
Loss depois do treino: 0.003461
Acurácia antes do treino: 0.500000
Acurácia depois do treino: 1.000000
Status: passou
```

Esse resultado não significa que a rede já está pronta para o MNIST, mas confirma que o fluxo completo de aprendizado funciona em um cenário controlado.

### Etapa 9 - Otimização

A preencher.

### Etapa 10 - Experimentos Comparativos

A preencher.

### Etapa 11 - Resultados

A preencher.

### Etapa 12 - README Final

A preencher.
