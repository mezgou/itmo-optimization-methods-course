# Датасеты

Лаборатории 3-4 используют CSV с последним столбцом `target`.

## Локальные файлы

- `data/first_dataset.csv` — d1, 600 строк, 2 признака, бинарный target.
- `data/second_dataset.csv` — d2, 600 строк, 4 признака, бинарный target.

Файлы маленькие и коммитятся в репозиторий.

## Google Drive download

Для защиты добавлена утилита:

```python
optlib.download(file_id, "data/third_dataset.csv")
```

CLI:

```powershell
uv run python scripts/download_dataset.py <file_id> <dest>
```

Известные id:

- d1: `16VzwtrZRP9QScMdueb3BK0fvZtP8a6sH`;
- d2: `13npPTtGGiKyug7fEKyvaZZbFavFkhGLm`.

## Pipeline

`prepare_dataset(path)` выполняет:

1. загрузку CSV;
2. стратифицированное 80/20 разбиение;
3. стандартизацию признаков по train;
4. применение той же стандартизации к test.

`train_binary_classifier` обучает `MLPClassifier`, а `evaluate(model, path,
standardizer)` прогоняет уже обученную модель на target-last CSV. Это же
используется для закрытого d3.

Для экспериментов доступны отдельные строительные блоки:

- `stratified_split(features, targets)` — детерминированное стратифицированное
  разбиение;
- `fit_standardizer(features)` — параметры стандартизации, обученные только на
  train;
- `run_lab3_experiment(paths)` — табличный прогон Adam/HeavyBall по d1/d2;
- `BinaryDatasetModel.save(path)` и `load_binary_dataset_model(path)` —
  сохранение весов MLP вместе с mean/std для live-прогона d3.

`train_binary_classifier` также принимает параметры `activation`,
`initialization`, `l2` и `schedule`, поэтому в лабораторной 4 тот же pipeline
используется для ablation по регуляризации, инициализации и расписаниям
learning rate.

## Метрики

Основная метрика — F1. Также считаются accuracy, precision, recall и confusion
matrix. Для d1/d2 используется бинарный F1; multiclass d3 будет расширен в S8,
если закрытый набор окажется многоклассовым.
