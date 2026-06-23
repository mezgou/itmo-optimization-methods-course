# Датасеты

Модуль `optlib.datasets` отвечает за CSV-загрузку, stratified split,
стандартизацию, обучение бинарной MLP и оценку закрытого датасета.

## Формат CSV

CSV должен содержать заголовок. Последний столбец считается target, остальные
столбцы - признаки.

Пример:

```text
feature_0,feature_1,target
0.12,-1.3,0
1.42,0.8,1
```

Загрузка:

```python
features, targets = optlib.load_csv("data/first_dataset.csv")
```

## Открытые данные

В репозитории лежат:

- `data/first_dataset.csv` - бинарная классификация, 2 признака;
- `data/second_dataset.csv` - бинарная классификация, 4 признака.

Google Drive id также доступны как константы:

- `optlib.FIRST_DATASET_ID`;
- `optlib.SECOND_DATASET_ID`.

## Загрузка по Google Drive id

```powershell
uv run python scripts/download_dataset.py <file_id> data/third_dataset.csv
```

или из Python:

```python
optlib.download("<file_id>", "data/third_dataset.csv")
```

Для загрузки нужен extra `experiments`, потому что он устанавливает `gdown`.

## Разбиение и стандартизация

Используется deterministic stratified split:

```python
split = optlib.load_dataset("data/first_dataset.csv", test_size=0.2, seed=42)
```

Стандартизация обучается только на train:

$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_{ij}
$$

$$
\sigma_j = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (x_{ij} - \mu_j)^2}
$$

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

Если $\sigma_j$ слишком мала, она заменяется на `1.0`.

## Метрики

Для бинарной классификации:

$$
precision = \frac{TP}{TP + FP}
$$

$$
recall = \frac{TP}{TP + FN}
$$

$$
F1 = \frac{2 precision \cdot recall}{precision + recall}
$$

API:

```python
metrics = optlib.binary_metrics(targets, probabilities)
```

Возвращаются `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`.

## Обучение модели на CSV

```python
result = optlib.train_binary_dataset(
    "data/first_dataset.csv",
    hidden_dim=12,
    method="adam",
    learning_rate=0.03,
    l2=1e-4,
)

print(result.test_metrics["f1"])
model_path = result.model.save("artifacts/d1_model.npz")
```

`DatasetTrainingResult` содержит модель, split, train metrics и test metrics.

## Закрытый d3

Закрытый CSV можно оценить двумя способами.

Если формат совместим с сохраненной бинарной моделью:

```python
metrics = optlib.evaluate_saved_model("artifacts/d1_model.npz", "data/third_dataset.csv")
```

Если нужно обучить pipeline заново и учесть возможный multiclass target:

```python
score = optlib.weighted_f1_score(
    {
        "d1": "data/first_dataset.csv",
        "d2": "data/second_dataset.csv",
        "d3": "data/third_dataset.csv",
    }
)
```

Взвешенная оценка:

$$
score = 0.3 F1_{d1} + 0.3 F1_{d2} + 0.4 F1_{d3}
$$

Если `d3` имеет больше двух классов, `train_dataset_score` использует
one-vs-rest схему и возвращает macro-F1.
