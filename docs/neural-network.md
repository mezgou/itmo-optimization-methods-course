# Нейронная сеть

`optlib` содержит компактную C++ MLP для бинарной классификации. Сеть обучается
теми же first-order оптимизаторами, что и обычные objective-функции: MLP
возвращает loss и flat gradient, а оптимизатор обновляет вектор параметров.

## Архитектура

Используется один скрытый слой:

$$
x \to W_1 x + b_1 \to a \to W_2 a + b_2 \to p
$$

Вероятность класса 1:

$$
p = \sigma(W_2 a + b_2)
$$

Число параметров:

$$
N = h d + h + h + 1
$$

где $d$ - число признаков, $h$ - размер скрытого слоя.

Поддерживаемые activation:

- `relu`;
- `leaky_relu`;
- `sigmoid`;
- `tanh`.

Поддерживаемые initialization:

- `xavier`;
- `he`.

## Loss

Binary Cross-Entropy:

$$
L =
-\frac{1}{m}
\sum_{i=1}^{m}
y_i \log p_i + (1 - y_i) \log(1 - p_i)
$$

Для `sigmoid + BCE` дельта выходного слоя:

$$
\delta = p - y
$$

L2-регуляризация добавляется к loss и градиенту весов.

## Backprop

Для одного объекта:

$$
dW_2 = (p - y) a^\top,\quad db_2 = p - y
$$

$$
da = W_2^\top (p - y)
$$

$$
dW_1 = (da \odot a') x^\top,\quad db_1 = da \odot a'
$$

Градиенты накапливаются по batch и делятся на число объектов.

## Native API

```python
count = optlib.BinaryMlpParameterCount(input_dim=4, hidden_dim=12)
params = optlib.InitializeBinaryMlpParameters(4, 12, activation="tanh")
loss = optlib.BinaryMlpLossAndGradient(params, x, y, 4, 12)
probabilities = optlib.BinaryMlpPredictProba(params, x, 4, 12)
f1 = optlib.BinaryF1Score(probabilities, y)
```

Полный training wrapper:

```python
result = optlib.TrainBinaryMlp(
    features,
    targets,
    hidden_dim=12,
    method="adam",
    learning_rate=0.03,
)
```

## Python classifier

```python
model = optlib.MLPClassifier(
    hidden_dim=12,
    activation="tanh",
    method="adam",
    learning_rate=0.03,
)
model.fit(features, targets)

probabilities = model.predict_proba(features)
labels = model.predict(features)
score = model.score(features, targets)
```

`score` возвращает F1. Параметры можно сохранить через
`BinaryDatasetModel.save`, если используется pipeline из `datasets.py`.

## Проверки

Корректность backprop проверяется gradient checking:

$$
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}
$$

Отдельно проверяется обучение XOR и воспроизводимость при фиксированном seed.
