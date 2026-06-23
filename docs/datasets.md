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
используется для закрытого d3, если его число признаков совпадает с сохранённой
binary-моделью. При несовпадении размерности `BinaryDatasetModel.transform`
выдаёт явную ошибку вместо неявного broadcasting.

Для экспериментов доступны отдельные строительные блоки:

- `stratified_split(features, targets)` — детерминированное стратифицированное
  разбиение;
- `fit_standardizer(features)` — параметры стандартизации, обученные только на
  train;
- `run_lab3_experiment(paths)` — табличный прогон Adam/HeavyBall по d1/d2;
- `BinaryDatasetModel.save(path)` и `load_binary_dataset_model(path)` —
  сохранение весов MLP вместе с mean/std для live-прогона d3.
- `evaluate_saved_model(model_path, dataset_path)` — загрузка сохранённой модели
  и оценка target-last CSV одним вызовом;
- `studies.weighted_f1_score({"d1": ..., "d2": ..., "d3": ...})` — итоговая
  формула `0.3F1(d1)+0.3F1(d2)+0.4F1(d3)` для доступных датасетов;
- `studies.train_dataset_score(path)` — binary F1 для бинарного CSV и
  one-vs-rest macro-F1 fallback для возможного multiclass d3.

`train_binary_classifier` также принимает параметры `activation`,
`initialization`, `l2` и `schedule`, поэтому в лабораторной 4 тот же pipeline
используется для ablation по регуляризации, инициализации и расписаниям
learning rate.

## Метрики

Основная метрика — F1. Также считаются accuracy, precision, recall и confusion
matrix. Для d1/d2 используется бинарный F1. Для возможного multiclass d3 Python
study layer использует one-vs-rest поверх той же binary MLP и считает macro-F1;
это отражено в итоговых ячейках `third_lab.ipynb` и `fourth_lab.ipynb`.
