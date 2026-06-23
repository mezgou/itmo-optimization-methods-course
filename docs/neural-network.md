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
hidden_dim * input_dim + hidden_dim + hidden_dim + 1
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
L = - mean[y log(p) + (1 - y) log(1 - p)]
```

Вероятность `p` считается стабильной sigmoid-функцией. В backprop для
`sigmoid + BCE` выходная дельта упрощается:

```text
delta = p - y
```

Опциональный L2 штраф добавляется к loss и градиенту всех весовых параметров.

## Метрики

В S6 реализован `BinaryF1Score`. Python wrapper `MLPClassifier.score` возвращает
F1, потому что именно F1 используется в критериях лабораторий 3-4.

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
