# Нейронная сеть

S6 добавляет минимальное C++ ядро для бинарной MLP-классификации. Этот слой
нужен для лабораторий 3-4 и специально связан с уже реализованными
оптимизаторами через плоский вектор параметров.

## Архитектура

Текущая модель:

```text
x -> Dense(input_dim, hidden_dim) -> activation -> Dense(hidden_dim, 1) -> sigmoid
```

Параметры хранятся в одном `Vector`:

```text
[W1, b1, W2, b2]
```

Размер:

```text
parameter_count = hidden_dim * input_dim + hidden_dim + hidden_dim + 1
```

Такой формат позволяет передать сеть в `MinimizeFirstOrder`: objective
возвращает BCE loss, а gradient callback заполняет градиент backprop по тому же
плоскому layout.

## Активации

Поддержаны:

- ReLU;
- LeakyReLU;
- Sigmoid;
- Tanh.

Для скрытого слоя в тестах и baseline используется `tanh`, потому что он хорошо
работает на XOR и небольших нормализованных табличных данных.

## Loss и backprop

Для бинарной классификации используется Binary Cross-Entropy:

```text
L = -(1/m) * sum_i [y_i*log(p_i) + (1 - y_i)*log(1 - p_i)]
```

Вероятность `p` считается стабильной sigmoid-функцией. В backprop для
`sigmoid + BCE` выходная дельта упрощается:

```text
delta = p - y
```

Опциональный L2 штраф добавляется к loss и градиенту всех весовых параметров.

Для одного объекта с входом `x`, скрытым слоем `a = activation(W1 x + b1)` и
логитом `z = W2 a + b2` backprop использует:

```text
dW2 = (p - y) * a^T
db2 = p - y
da  = W2^T * (p - y)
dW1 = (da * activation'(W1*x + b1)) * x^T
db1 = da * activation'(W1*x + b1)
```

В реализации все `dW/db` накапливаются по batch и делятся на число объектов.

## Метрики

В S6 реализован `BinaryF1Score`. Python wrapper `MLPClassifier.score` возвращает
F1, потому что именно F1 используется в критериях лабораторий 3-4.

Для d1/d2 метрика бинарная. Для закрытого d3 структура заранее неизвестна:
если target окажется multiclass, `studies.train_dataset_score` обучает
one-vs-rest набор binary MLP и возвращает macro-F1. Это не меняет C++ ядро:
оно остаётся быстрым бинарным building block, а multiclass fallback собирается
на Python-уровне для отчёта и защиты.

## Python API

Низкоуровневые функции:

```python
params = optlib.InitializeBinaryMlpParameters(2, 8, seed=42)
loss_grad = optlib.BinaryMlpLossAndGradient(params, X, y, 2, 8)
proba = optlib.BinaryMlpPredictProba(params, X, 2, 8)
```

Высокоуровневый wrapper:

```python
model = optlib.MLPClassifier(hidden_dim=8, learning_rate=0.05).fit(X, y)
pred = model.predict(X)
f1 = model.score(X, y)
```

## Проверка градиента

`tests/cpp/TestNeuralNetwork.cpp` сравнивает backprop-градиент с центральной
конечной разностью по всем параметрам маленькой XOR-сети. Это защищает flat
offset layout и формулы backprop от смещений индексов.

## Лаборатория 3

S7 добавляет Python pipeline вокруг C++ MLP:

- `MLPClassifier.fit/predict/predict_proba/score`;
- `train_binary_classifier(path, method=...)`;
- `evaluate(model, path, standardizer)`;
- стратифицированное 80/20 разбиение и стандартизацию признаков.

Для d1/d2 baseline обучается минимум двумя оптимизаторами: Adam и HeavyBall.

`TrainBinaryMlp` и `MLPClassifier` принимают LR schedules (`constant`, `step`,
`exponential`, `cosine`) и warmup-параметры. При `log_trajectory=True`
wrapper сохраняет `optimizer_result_`, из которого notebook строит кривые BCE
по итерациям без повторной реализации обучения в Python.

## Лаборатория 4

В `fourth_lab.ipynb` проверяются:

- все first-order оптимизаторы из лабораторий 1-2 на d1/d2;
- устойчивость по seed для Adam, HeavyBall и Nesterov;
- расписания `constant`, `step`, `exponential`, `cosine`, `cosine+warmup`;
- L2 ablation и initialization ablation (`xavier` против `he`);
- внешний baseline через `sklearn.MLPClassifier` и optional PyTorch;
- сохранение/загрузка финальной binary-модели и weighted F1 для d1/d2/d3.

Финальный конфиг в ноутбуке фиксируется явно:

```text
hidden_dim=16, method=adam, learning_rate=0.03,
max_iter=5000, activation=tanh, initialization=xavier, l2=1e-4
```
