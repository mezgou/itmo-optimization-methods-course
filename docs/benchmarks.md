# Бенчмарки и отчеты

Бенчмарки в проекте нужны для сравнения качества, скорости и устойчивости
методов. Основные метрики фиксируют не только финальное значение функции, но и
цену получения результата.

## Метрики оптимизации

Для каждого запуска собираются:

- `value`;
- `gradient_norm`;
- `iterations`;
- `function_evaluations`;
- `gradient_evaluations`;
- `hessian_evaluations`;
- `wall_ms`;
- `converged`;
- `distance_to_minimum`, если минимум известен.

Python helper:

```python
objective = optlib.get_objective("rastrigin", dimension=10)
rows = optlib.compare_methods(
    objective,
    x0,
    ["adam", "lbfgs", "nelder_mead"],
    max_iter=1000,
    log_trajectory=False,
)
```

Для 2D задач можно включать `log_trajectory=True`, чтобы строить контурные
карты. Для ND сравнений лучше отключать траекторию, иначе память и время
искажают измерения.

## Native wrappers

Для честного wall-clock сравнения harness использует native wrappers:

```python
optlib.MinimizeBenchmarkFunction("rastrigin", x0, method="adam")
optlib.MinimizeBenchmarkFunctionSecondOrder("ackley", x0, method="lbfgs")
optlib.MinimizeBenchmarkFunctionZeroOrder("desmos_surface", x0, method="powell")
```

Так стоимость Python callback не смешивается со стоимостью C++ оптимизатора.

## Внешние baseline

SciPy используется как optional reference:

```python
reference = optlib.scipy_minimize(objective, x0, method="BFGS")
```

Если SciPy не установлен, функция возвращает `None`.

Для MLP доступны optional baseline:

```python
sk = optlib.sklearn_mlp_baseline("data/first_dataset.csv")
torch_row = optlib.torch_mlp_baseline("data/first_dataset.csv")
```

Они нужны только для сравнительных отчетов и не являются зависимостями C++ ядра.

## C++ microbenchmark

Release microbenchmark:

```powershell
cmake -B build -DOPTLIB_BUILD_BENCHMARKS=ON
cmake --build build --config Release
.\build\Release\optlib_core_bench.exe
```

Он измеряет базовые операции ядра: `Dot`, `Gemv`, конечные разности и autograd.

## Ноутбуки

Отчеты находятся в `notebooks/`:

- `first_lab.ipynb` - конечные разности, autograd, first-order методы,
  траектории Rosenbrock, сравнение с SciPy.
- `second_lab.ipynb` - first-order, second-order и zero-order методы на наборе
  benchmark-функций, включая DesmosSurface.
- `third_lab.ipynb` - обучение MLP на d1/d2, F1, confusion matrix, граница
  решений для 2D данных.
- `fourth_lab.ipynb` - сравнение оптимизаторов MLP, schedules, stability по
  seed, L2/init ablation, sklearn/PyTorch baseline, d3-ready evaluation.

Генераторы ноутбуков:

```powershell
uv run python notebooks/_build_first_lab.py
uv run python notebooks/_build_second_lab.py
uv run python notebooks/_build_third_lab.py
uv run python notebooks/_build_fourth_lab.py
```

## Воспроизводимость

В отчетах фиксируются:

- seed;
- стартовые точки;
- размерность задачи;
- параметры оптимизатора;
- включение или отключение траектории;
- версия `optlib`.

Для сравнений важно запускать все методы на одинаковых входах и явно показывать
таблицу параметров.
