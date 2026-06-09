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

A preencher.

### Etapa 6 - SGD

A preencher.

### Etapa 7 - Mini-Batch Training

A preencher.

### Etapa 8 - Treinamento Inicial

A preencher.

### Etapa 9 - Otimização

A preencher.

### Etapa 10 - Experimentos Comparativos

A preencher.

### Etapa 11 - Resultados

A preencher.

### Etapa 12 - README Final

A preencher.
