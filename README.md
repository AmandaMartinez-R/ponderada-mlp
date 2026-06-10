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

Nesta etapa, preparei a rede para o processo de otimização dos hiperparâmetros. Antes, o método `fit` registrava apenas a loss de treino. Isso era pouco para comparar configurações, porque uma loss menor nem sempre mostra sozinha se a rede está classificando melhor ou se está generalizando bem.

Adicionei ao histórico de treinamento as métricas de `loss`, `accuracy`, `val_loss` e `val_accuracy`. Com isso, fica mais fácil acompanhar a evolução da rede ao longo das épocas e comparar escolhas como learning rate, batch size, quantidade de neurônios e quantidade de camadas.

Também adicionei validação opcional no `fit`, usando `X_val` e `y_val`. Mantive treino e validação separados para evitar misturar os dados usados na atualização dos pesos com os dados usados apenas para avaliação.

Entendi que otimização não é apenas testar valores aleatórios. A ideia é observar o comportamento da loss e da acurácia: se a loss oscila muito, talvez o learning rate esteja alto; se a loss cai devagar demais, talvez o learning rate esteja baixo; se a acurácia de treino sobe muito e a validação não acompanha, pode haver overfitting.

Resultados do teste com validação:

```text
Loss de treino: 0.630900 -> 0.009466
Acurácia de treino: 0.500000 -> 1.000000
Loss de validação: 0.649124 -> 0.010083
Acurácia de validação: 0.500000 -> 1.000000
Status: passou
```

Também mantive compatibilidade com chamadas antigas do `fit`, sem validação. Quando nenhum conjunto de validação é informado, as listas `val_loss` e `val_accuracy` ficam vazias.

### Etapa 10 - Experimentos Comparativos

Nesta etapa, criei o notebook `notebooks/experimentos.ipynb` para organizar os primeiros experimentos comparativos. Antes de usar o MNIST, comparei duas configurações em um dataset sintético simples, porque isso permite validar o fluxo de experimentação em um ambiente controlado.

Comparei duas arquiteturas diferentes mantendo o mesmo problema, a mesma divisão de treino e validação, o mesmo batch size e a mesma quantidade de épocas. A diferença principal foi o tamanho da rede e o learning rate.

Configurações testadas:

```text
Configuração A:
Arquitetura: 2 -> 8 -> 4 -> 2
Learning rate: 0.1
Batch size: 4
Épocas: 40

Configuração B:
Arquitetura: 2 -> 16 -> 8 -> 2
Learning rate: 0.05
Batch size: 4
Épocas: 40
```

Resultados obtidos:

```text
Configuração A:
Loss de treino: 0.009466
Acurácia de treino: 1.000000
Loss de validação: 0.010083
Acurácia de validação: 1.000000

Configuração B:
Loss de treino: 0.018731
Acurácia de treino: 1.000000
Loss de validação: 0.020407
Acurácia de validação: 1.000000
```

As duas configurações conseguiram aprender o dataset sintético e chegaram a 100% de acurácia em treino e validação. A Configuração A terminou com loss menor, mesmo sendo menor, provavelmente porque o problema era simples e o learning rate maior ajudou a convergir mais rápido nesse cenário.

Esse resultado ainda não define a melhor configuração para o MNIST. Ele apenas confirma que o processo de comparar arquiteturas, registrar métricas e analisar resultados já está funcionando. No MNIST, espero que as diferenças entre configurações fiquem mais relevantes.

### Etapa 11 - Resultados

Nesta etapa, gerei os primeiros arquivos de resultados na pasta `results/`. Esses resultados ainda são preliminares, porque foram produzidos com o dataset sintético usado nos experimentos comparativos, e não com o MNIST.

Mesmo assim, achei importante salvar os artefatos desde já, porque isso valida o fluxo de documentação dos resultados: uma tabela com as métricas e gráficos simples para visualizar a comparação entre configurações.

Arquivos gerados:

```text
results/experimentos_comparativos.csv
results/loss_comparativo.svg
results/accuracy_comparativo.svg
```

O arquivo CSV registra as métricas finais das duas configurações testadas: arquitetura, learning rate, batch size, épocas, loss de treino, acurácia de treino, loss de validação e acurácia de validação.

Também gerei dois gráficos em SVG:

```text
loss_comparativo.svg: compara loss final de treino e validação
accuracy_comparativo.svg: compara acurácia final de treino e validação
```

Como o ambiente atual não tinha `matplotlib` disponível, optei por gerar os gráficos diretamente em SVG. Essa escolha evita instalar dependências nesta etapa e ainda produz arquivos que podem ser abertos no navegador ou visualizados no GitHub.

Resultados registrados:

```text
Configuração A:
Loss de treino: 0.009466
Acurácia de treino: 1.000000
Loss de validação: 0.010083
Acurácia de validação: 1.000000

Configuração B:
Loss de treino: 0.018731
Acurácia de treino: 1.000000
Loss de validação: 0.020407
Acurácia de validação: 1.000000
```

Esses resultados confirmam que o processo de gerar e salvar artefatos funciona. Na etapa final com MNIST, essa mesma lógica será usada para salvar curvas reais de loss e acurácia ao longo das épocas.

### Etapa 12A - Treinamento no MNIST

Nesta etapa, preparei o projeto para executar o treinamento real no MNIST. Até aqui, os testes tinham sido feitos com um dataset sintético, o que foi útil para validar a implementação, mas ainda não atendia ao requisito principal da atividade: treinar no MNIST e buscar pelo menos 92% de acurácia no conjunto de teste.

Atualizei o arquivo `requirements.txt` com as dependências necessárias para essa fase:

```text
numpy
tensorflow
matplotlib
notebook
```

A escolha de incluir `tensorflow` foi feita apenas para carregar o dataset MNIST usando `tensorflow.keras.datasets.mnist`. O treinamento da rede continua sendo feito pela implementação própria em NumPy, sem usar TensorFlow para montar modelo, calcular gradientes ou atualizar pesos.

Também atualizei o notebook `notebooks/experimentos.ipynb` com uma seção específica para MNIST. Essa seção faz:

```text
Carregamento do MNIST via Keras
Normalização dos pixels para o intervalo [0, 1]
Transformação das imagens 28x28 em vetores de 784 posições
Separação de 10.000 imagens para validação
Treinamento de duas configurações diferentes
Avaliação no conjunto de teste
Salvamento de CSV e gráficos em results/
```

Configurações preparadas para o MNIST:

```text
MNIST A:
Arquitetura: 784 -> 128 -> 64 -> 10
Learning rate: 0.1
Batch size: 128
Épocas: 12

MNIST B:
Arquitetura: 784 -> 256 -> 128 -> 10
Learning rate: 0.05
Batch size: 128
Épocas: 12
```

Ainda não executei o treinamento MNIST neste ambiente, porque o runtime atual não tinha `tensorflow`, `keras` nem `matplotlib` instalados. A execução deve ser feita depois de instalar as dependências com `pip install -r requirements.txt`.

Resultados esperados após executar a seção MNIST:

```text
results/mnist_experimentos.csv
results/mnist_loss.png
results/mnist_accuracy.png
```

Essa etapa deixa o projeto pronto para o teste principal. Depois que o notebook for executado, preciso copiar os resultados reais para o README final e verificar se alguma configuração atingiu a meta de 92% de acurácia no teste.

### Etapa 12 - README Final

A preencher.
