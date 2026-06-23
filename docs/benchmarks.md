# Бенчмарки

Этот раздел фиксирует способ воспроизведения измерений. Метрики строятся так,
чтобы сравнивать не только итоговое значение, но и цену результата: число
вычислений функции, градиента, гессиана и wall-clock время.

## Лабораторная 1

Мини-бенчмарки находятся в двух местах:

- `benchmarks/cpp/CoreBench.cpp` — C++ микробенч линейной алгебры,
  конечных разностей и autograd.
- `notebooks/first_lab.ipynb` — Python-измерения времени градиентов и сравнение
  с `scipy.optimize.minimize`.

Запуск C++ бенча после сборки:

```powershell
cmake -B build -DOPTLIB_BUILD_TESTS=ON
cmake --build build --config Release
.\build\Release\optlib_core_bench.exe
```

Запуск ноутбука с экспериментальными зависимостями:

```powershell
uv sync --extra experiments --extra dev
uv run jupyter notebook notebooks/first_lab.ipynb
```

## Что сравнивается

- аналитический градиент Розенброка;
- multidual autograd;
- центральная и пятиточечная конечные разности;
- `optlib.MinimizeRosenbrock` против `scipy.optimize.minimize` как внешнего
  эталона.

## Лабораторная 2

Для второй лабораторной используются функции:

- Rosenbrock: гладкая узкая долина;
- Rastrigin: мультимодальная ND-функция;
- Himmelblau: 2D-гладкая функция с несколькими минимумами;
- Ackley: ND-функция с плато;
- DesmosSurface: разрывная black-box поверхность.

Методы сравниваются по единой строке результата:

- `value`;
- `gradient_norm`, если градиент определён;
- `iterations`;
- `function_evaluations`;
- `gradient_evaluations`;
- `hessian_evaluations`;
- `distance_to_minimum`, если минимум известен;
- `converged`.

Python-харнесс:

```python
objective = optlib.get_objective("rastrigin", dimension=10)
rows = optlib.compare_methods(
    objective,
    x0,
    ["adam", "lbfgs", "nelder_mead", "powell"],
    max_iter=1000,
    log_trajectory=False,
)
```

Для измерений `optlib` harness использует native wrappers, а не Python
callbacks:

```python
optlib.MinimizeBenchmarkFunction("rastrigin", x0, method="adam")
optlib.MinimizeBenchmarkFunctionSecondOrder("ackley", x0, method="lbfgs")
optlib.MinimizeBenchmarkFunctionZeroOrder("desmos_surface", x0, method="powell")
```

Так wall-clock сравнение не смешивает стоимость C++ оптимизатора со стоимостью
перехода в Python на каждом вычислении функции.

SciPy запускается только как экспериментальная зависимость:

```python
reference = optlib.scipy_minimize(objective, x0, method="BFGS")
```

Если SciPy не установлен, функция возвращает `None`, а ноутбук продолжает
работать только с `optlib`.

## Воспроизводимость

Для multistart используется фиксированный seed. Размерности для ND-функций:

```text
n in {2, 10, 50, 100}
```

Для 2D-функций сохраняются траектории, чтобы строить контурные карты. Для
масштабных ND-сравнений `log_trajectory=False`, иначе запись всех точек
искажает измерение памяти и времени.

## Лаборатория 4

Для нейросети основная метрика — F1 на test split. Сравнение строится по
таблицам:

- optimizer: GD, HeavyBall, Nesterov, Adam, RMSProp, Adagrad;
- stability: F1/loss по нескольким seed для Adam, HeavyBall, Nesterov;
- schedule: constant, step, exponential, cosine и cosine+warmup;
- регуляризация: набор L2 значений;
- initialization: Xavier против He;
- внешний baseline: `sklearn.MLPClassifier` и optional PyTorch baseline.

Запуск:

```python
rows = optlib.compare_nn_optimizers(
    "data/first_dataset.csv",
    methods=["adam", "heavy_ball", "rmsprop"],
    schedule="cosine",
)
```

Для d3 есть два режима. Если закрытый CSV совместим по числу признаков с
сохранённой binary-моделью, используется `evaluate(model, path)` с тем же
preprocessing. Если структура другая или d3 окажется multiclass, итоговая
таблица использует `studies.weighted_f1_score` и one-vs-rest macro-F1 fallback.

Главный воспроизводимый отчет:

```powershell
uv sync --extra experiments --extra dev
uv run jupyter notebook notebooks/fourth_lab.ipynb
```

## Контрольный запуск S9

Финальный C++ микробенчмарк запускался из Release-сборки на Windows через
`std::chrono::steady_clock`. В `CoreBench.cpp` зафиксированы `DIMENSION = 256`
и `REPEATS = 25`, ниже приведена медиана по повторам:

```powershell
.\build\Release\optlib_core_bench.exe
```

| benchmark | median_us |
| --- | ---: |
| `Dot256x1000` | 0.6 |
| `Gemv256` | 55.9 |
| `CentralGradient256` | 246.9 |
| `AutogradGradient256` | 1710.8 |

Эти числа нужны как быстрая проверка производительности ядра. Для отчетных
сравнений с SciPy/sklearn следует запускать ноутбуки, потому что они фиксируют
не только время, но и итоговое качество, число вычислений и устойчивость метода.
