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

Commit sugerido:

```text
feat: implementa ReLU e derivada
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

Commit sugerido:

```text
feat: implementa softmax e cross entropy
```

### Etapa 3 - Inicialização dos Pesos

A preencher.

### Etapa 4 - Forward Pass

A preencher.

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
