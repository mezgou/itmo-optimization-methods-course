# Справочник API

Пакет импортируется как `optlib`. Нативные функции из `_optlib` сохраняют
PascalCase имена C++ API. Python helpers используют snake_case.

```python
import optlib

print(optlib.__version__)
print(optlib.Version())
```

## Линейная алгебра

- `Dot(left, right)` - скалярное произведение.
- `Norm2(values)` - евклидова норма.
- `NormInf(values)` - бесконечная норма.
- `Axpy(alpha, x, y)` - `y + alpha * x`.
- `Gemv(matrix, vector)` - матрица на вектор.
- `Gemm(left, right)` - матрица на матрицу.

## Дифференцирование

- `NumericGradient(function, x, scheme="central", step=0.0)`;
- `RosenbrockNumericalGradient(x, scheme="central", step=0.0)`;
- `RosenbrockAutogradGradient(x)`.

Схемы: `"forward"`, `"central"`, `"five_point"`.

## Benchmark-функции

Специализированные функции:

- `RosenbrockValue`, `RosenbrockGradient`, `RosenbrockHessian`;
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

- `BenchmarkFunctionInfo(name, dimension=2)`;
- `BenchmarkFunctionValue(name, x, scale=1.0)`;
- `BenchmarkFunctionGradient(name, x)`;
- `BenchmarkFunctionHessian(name, x)`.

Python registry:

```python
names = optlib.list_objectives()
objective = optlib.get_objective("rosenbrock", dimension=10)
```

## Оптимизаторы

First-order:

```python
result = optlib.Minimize(value, gradient, x0, method="adam")
result = optlib.MinimizeRosenbrock(x0, method="heavy_ball")
result = optlib.MinimizeBenchmarkFunction("rastrigin", x0, method="rmsprop")
```

Методы: `"gradient_descent"`, `"heavy_ball"`, `"nesterov"`, `"adam"`,
`"rmsprop"`, `"adagrad"`.

Second-order:

```python
result = optlib.MinimizeSecondOrder(value, gradient, x0, method="lbfgs")
result = optlib.MinimizeRosenbrockSecondOrder(x0, method="bfgs")
result = optlib.MinimizeBenchmarkFunctionSecondOrder("ackley", x0, method="newton")
```

Методы: `"newton"`, `"bfgs"`, `"lbfgs"`.

Zero-order:

```python
result = optlib.MinimizeZeroOrder(value, x0, method="powell")
result = optlib.MinimizeRosenbrockZeroOrder(x0, method="nelder_mead")
result = optlib.MinimizeBenchmarkFunctionZeroOrder("desmos_surface", x0)
```

Методы: `"nelder_mead"`, `"powell"`, `"coordinate_search"`.

Learning-rate helper:

```python
lr = optlib.LearningRateAt(10, schedule="step", initial_learning_rate=0.1)
```

Schedules: `"constant"`, `"step"`, `"exponential"`, `"cosine"`.

## Результат оптимизации

Оптимизаторы возвращают `dict`:

- `x`;
- `value`;
- `gradient_norm`;
- `iterations`;
- `function_evaluations`;
- `gradient_evaluations`;
- `hessian_evaluations`;
- `converged`;
- `trajectory`.

`trajectory` содержит `x`, `f`, `grad_norm`, `time_ms`.

## MLP

Native API:

- `BinaryMlpParameterCount(input_dim, hidden_dim)`;
- `InitializeBinaryMlpParameters(input_dim, hidden_dim, activation="tanh", initialization="xavier")`;
- `BinaryMlpPredictProba(parameters, features, input_dim, hidden_dim, activation="tanh")`;
- `BinaryMlpLossAndGradient(parameters, features, targets, input_dim, hidden_dim, activation="tanh", l2=0.0)`;
- `BinaryF1Score(probabilities, targets, threshold=0.5)`;
- `TrainBinaryMlp(features, targets, hidden_dim=8, method="adam")`.

Python wrapper:

```python
model = optlib.MLPClassifier(hidden_dim=12, method="adam")
model.fit(features, targets)
probabilities = model.predict_proba(features)
labels = model.predict(features)
f1 = model.score(features, targets)
```

## Данные и study helpers

CSV и preprocessing:

- `load_csv(path)`;
- `load_dataset(path, test_size=0.2, seed=42)`;
- `prepare_dataset(path, test_size=0.2, seed=42)`;
- `stratified_split(features, targets, test_size=0.2, seed=42)`;
- `fit_standardizer(features)`;
- `download(file_id, dest)`.

Обучение и оценка:

- `train_binary_dataset(path, ...)`;
- `train_binary_classifier(path, ...)`;
- `evaluate(model, path, standardizer=None)`;
- `evaluate_saved_model(model_path, dataset_path)`;
- `binary_metrics(targets, probabilities, threshold=0.5)`;
- `classification_metrics(targets, predictions)`.

Сравнительные исследования:

- `compare_methods(objective, x0, methods, ...)`;
- `multistart_compare(objective, methods, starts=..., ...)`;
- `run_method(objective, x0, method, ...)`;
- `result_summary(method, result, objective=None, wall_ms=None)`;
- `scipy_minimize(objective, x0, method="BFGS")`;
- `compare_nn_optimizers(path, ...)`;
- `regularization_ablation(path, ...)`;
- `initialization_ablation(path, ...)`;
- `optimizer_stability(path, ...)`;
- `train_dataset_score(path, ...)`;
- `weighted_f1_score(paths, weights=None, ...)`;
- `sklearn_mlp_baseline(path, ...)`;
- `torch_mlp_baseline(path, ...)`.

Optional baseline функции возвращают `None`, если соответствующая внешняя
библиотека не установлена.

## Plotting helpers

Функции из `optlib.plotting` используются ноутбуками и доступны через submodule:

- `use_notebook_style()`;
- `objective_grid(objective, x_range, y_range, points=...)`;
- `plot_contours(...)`;
- `plot_trajectories(...)`;
- `plot_convergence(...)`;
- `plot_surface3d(...)`;
- `plot_confusion_matrix(...)`;
- `plot_decision_boundary(...)`;
- `bar_comparison(...)`.

Они не являются частью C++ ядра и не влияют на пакетный runtime без extra
`experiments`.

## C++ header surface

Публичная C++ поверхность расположена в `src/optlib/core/`:

- `Vector`, `Matrix`;
- `Trajectory`, `OptimizeResult`, `StopCriteria`;
- `Dual`;
- `ComputeGradient`, `ComputeAutogradGradient`;
- benchmark-функции и `BenchmarkFunctionInfo`;
- `FirstOrderConfig`, `SecondOrderConfig`, `ZeroOrderConfig`;
- `MinimizeFirstOrder`, `MinimizeSecondOrder`, `MinimizeZeroOrder`;
- `LineSearchConfig`, `ArmijoLineSearch`, `WolfeLineSearch`;
- `LearningRateConfig`, `LearningRateAt`;
- `BinaryMlpConfig`, `BinaryMlpTrainingResult`, native train/predict/loss/F1 API.

Для внешнего пользователя рекомендуемый стабильный интерфейс - Python API.
