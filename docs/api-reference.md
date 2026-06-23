# Справочник API

Этот раздел фиксирует публичный Python API пакета `optlib`. C++ имена
сохраняются в PascalCase, потому что Python-слой реэкспортирует функции из
расширения `_optlib`; вспомогательные экспериментальные обертки в
`python/optlib/*.py` используют обычный snake_case.

## Импорт и версия

```python
import optlib

print(optlib.__version__)
print(optlib.Version())
```

`optlib` собирается как Python-пакет с C++23-расширением `_optlib`. Горячие
нативные функции освобождают GIL. На вход ожидаются `numpy.ndarray` с
`float64`; C-contiguous массивы проходят без лишней копии, несовместимые входы
могут быть приведены pybind11.

## Линейная алгебра

Функции работают с одномерными или двумерными NumPy-массивами:

- `Dot(left, right)` - скалярное произведение.
- `Norm2(values)` - евклидова норма.
- `NormInf(values)` - бесконечная норма.
- `Axpy(alpha, x, y)` - возвращает `y + alpha * x`.
- `Gemv(matrix, vector)` - матрично-векторное произведение.
- `Gemm(left, right)` - матричное произведение.

В C++ ядре этим функциям соответствуют контейнеры `Vector` и `Matrix` в
`src/optlib/core/LinAlg.hpp`; Python API намеренно не раскрывает владение этими
контейнерами напрямую.

## Дифференцирование и функции

Базовые функции Розенброка:

- `RosenbrockValue(x)`
- `RosenbrockGradient(x)`
- `RosenbrockHessian(x)`
- `RosenbrockNumericalGradient(x, scheme="central", step=0.0)`
- `RosenbrockAutogradGradient(x)`

Для произвольной Python-функции доступна численная производная:

```python
gradient = optlib.NumericGradient(objective, x, scheme="five_point")
```

Поддерживаемые схемы: `"forward"`, `"central"`, `"five_point"`. Если `step=0`,
ядро выбирает шаг по машинному epsilon и масштабу координаты.

Benchmark-функции лабораторной 2:

- `RastriginValue`, `RastriginGradient`, `RastriginHessian`,
  `RastriginAutogradGradient`;
- `HimmelblauValue`, `HimmelblauGradient`, `HimmelblauHessian`,
  `HimmelblauAutogradGradient`;
- `AckleyValue`, `AckleyGradient`, `AckleyAutogradGradient`;
- `BealeValue`, `BealeGradient`, `BealeHessian`;
- `BoothValue`, `BoothGradient`, `BoothHessian`;
- `StyblinskiTangValue`, `StyblinskiTangGradient`, `StyblinskiTangHessian`;
- `DesmosSurfaceValue`, `DesmosSurfaceNumericalGradient`.

Унифицированный доступ:

- `BenchmarkFunctionInfo(name, dimension=2)`
- `BenchmarkFunctionValue(name, x, scale=1.0)`
- `BenchmarkFunctionGradient(name, x)`
- `BenchmarkFunctionHessian(name, x)`

Python-обертки `get_objective(name, dimension)` и `list_objectives()` находятся
в `optlib.functions` и реэкспортируются из `optlib`.

## Оптимизаторы

Методы первого порядка:

- `Minimize(value_function, gradient_function, x0, method="adam", ...)`
- `MinimizeRosenbrock(x0, method="adam", ...)`
- `MinimizeBenchmarkFunction(function_name, x0, method="adam", ...)`

Поддерживаемые `method`: `"gradient_descent"`, `"heavy_ball"`, `"nesterov"`,
`"adam"`, `"rmsprop"`, `"adagrad"`.

Методы второго порядка:

- `MinimizeSecondOrder(value_function, gradient_function, x0, method="lbfgs", ...)`
- `MinimizeRosenbrockSecondOrder(x0, method="bfgs", ...)`
- `MinimizeBenchmarkFunctionSecondOrder(function_name, x0, method="lbfgs", ...)`

Поддерживаемые `method`: `"newton"`, `"bfgs"`, `"lbfgs"`. Для Newton можно
передать `hessian_function`; BFGS и L-BFGS используют градиенты и line search.

Методы нулевого порядка:

- `MinimizeZeroOrder(value_function, x0, method="nelder_mead", ...)`
- `MinimizeRosenbrockZeroOrder(x0, method="nelder_mead", ...)`
- `MinimizeBenchmarkFunctionZeroOrder(function_name, x0, method="nelder_mead", ...)`

Поддерживаемые `method`: `"nelder_mead"`, `"powell"`, `"coordinate_search"`.

Все оптимизаторы возвращают `dict`:

- `x` - найденная точка;
- `value` - значение функции;
- `gradient_norm` - норма градиента, если метод ее вычисляет;
- `iterations`;
- `function_evaluations`;
- `gradient_evaluations`;
- `hessian_evaluations`;
- `converged`;
- `trajectory` - словарь `x`, `f`, `grad_norm`, `time_ms`.

Для learning-rate schedules доступны:

- `LearningRateAt(iteration, schedule="constant", initial_learning_rate=...)`
- `LearningRateSchedule` enum из C++.

Строковые schedules: `"constant"`, `"step"`, `"exponential"`, `"cosine"`.

## Нейронная сеть

Нативные функции MLP:

- `BinaryMlpParameterCount(input_dim, hidden_dim)`
- `InitializeBinaryMlpParameters(input_dim, hidden_dim, activation="tanh", ...)`
- `BinaryMlpPredictProba(parameters, features, input_dim, hidden_dim, activation="tanh")`
- `BinaryMlpLossAndGradient(parameters, features, targets, input_dim, hidden_dim, ...)`
- `BinaryF1Score(probabilities, targets, threshold=0.5)`
- `TrainBinaryMlp(features, targets, hidden_dim=8, method="adam", ...)`

Высокоуровневый Python-класс:

```python
model = optlib.MLPClassifier(hidden_dim=8, method="adam")
model.fit(features, targets)
probabilities = model.predict_proba(features)
labels = model.predict(features)
```

Поддерживаемые activation: `"relu"`, `"leaky_relu"`, `"sigmoid"`, `"tanh"`.
Поддерживаемые initialization: `"xavier"`, `"he"`.

## Датасеты и исследования

Работа с d1/d2/d3:

- `load_csv(path)`
- `load_dataset(path)`
- `download(file_id, dest)`
- `stratified_indices(targets, test_size=0.2, seed=42)`
- `stratified_split(features, targets, test_size=0.2, seed=42)`
- `fit_standardizer(features)`
- `prepare_dataset(path, test_size=0.2, seed=42)`
- `train_binary_classifier(path, ...)`
- `train_binary_dataset(path, ...)`
- `evaluate(model, path, standardizer=None)`
- `evaluate_saved_model(model_path, dataset_path)`
- `run_lab3_experiment(path, ...)`
- `classification_metrics(predictions, targets)`
- `binary_metrics(probabilities, targets, threshold=0.5)`

Идентификаторы Google Drive:

- `FIRST_DATASET_ID`
- `SECOND_DATASET_ID`

Исследования лабораторной 4:

- `compare_nn_optimizers(path, methods=None, schedule="constant", ...)`
- `regularization_ablation(path, l2_values=None, ...)`
- `initialization_ablation(path, initializations=None, ...)`
- `optimizer_stability(path, methods=None, seeds=None, ...)`
- `sklearn_mlp_baseline(path, ...)`
- `torch_mlp_baseline(path, ...)`
- `train_dataset_score(path, ...)`
- `weighted_f1_score(paths, weights=None, ...)`

Если `scikit-learn` или `torch` не установлены, соответствующие baseline-функции
возвращают `None`.

`train_dataset_score` использует native binary MLP для бинарных CSV. Если в
target больше двух классов, функция обучает one-vs-rest набор binary MLP и
возвращает macro-F1; это нужно для закрытого d3 с заранее неизвестным числом
классов.

## Экспериментальные обертки

Для лабораторных используются дополнительные функции:

- `compare_methods(objective, x0, methods, ...)`
- `multistart_compare(objective, methods, ...)`
- `run_method(objective, x0, method, ...)`
- `result_summary(result, objective=None)`
- `scipy_minimize(objective, x0, method="BFGS")`
- `make_xor()`
- `make_two_moons()`
- `binary_mlp_loss_and_gradient(...)`

`scipy_minimize` и `sklearn_mlp_baseline` являются внешними эталонами для
экспериментов. Если соответствующая зависимость не установлена, обертка не
ломает основной пакет `optlib`.
