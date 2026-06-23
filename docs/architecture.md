# Архитектура

`optlib` состоит из трех слоев:

- C++23 ядро в `src/optlib/core/`;
- pybind11 bindings в `src/optlib/bindings/Module.cpp`;
- Python пакет в `python/optlib/`.

Слой C++ отвечает за вычислительно дорогие операции. Python слой отвечает за
экспериментальные сценарии, графики, загрузку данных и удобные wrappers.

## C++ ядро

Ключевые модули:

- `LinAlg.hpp` - плотные `Vector` и `Matrix`, `Dot`, `Norm2`, `NormInf`,
  `Axpy`, `Gemv`, `Gemm`.
- `Dual.hpp` - dual numbers для forward-mode autodiff.
- `Differentiation.hpp/.cpp` - конечные разности и autograd gradient.
- `functions/` - тестовые функции и registry benchmark-функций.
- `optimizers/FirstOrder.*` - Gradient Descent, HeavyBall, Nesterov, Adam,
  RMSProp, Adagrad.
- `optimizers/SecondOrder.*` - Newton, BFGS, L-BFGS и line search.
- `optimizers/ZeroOrder.*` - Nelder-Mead, Powell, coordinate search.
- `schedulers/LearningRate.*` - `constant`, `step`, `exponential`, `cosine`,
  warmup.
- `nn/NeuralNetwork.*` - бинарная MLP с плоским вектором параметров.
- `OptimizeResult.hpp` - общий результат оптимизации и траектория.

Ядро не зависит от Eigen, BLAS, Boost или CUDA. Внутренние контейнеры хранят
данные в contiguous `double` буферах.

## Граница pybind11

Bindings принимают `numpy.ndarray` и приводят входы к C-contiguous `float64`,
если это необходимо. Для результатов C++ создает собственное хранилище и
передает владение в NumPy через `py::capsule`.

Горячие C++ участки освобождают GIL. Если пользователь передает Python callback
для функции или градиента, GIL захватывается только на время этого callback.

## Python слой

`python/optlib/__init__.py` реэкспортирует публичный API из `_optlib` и helper
модулей:

- `functions.py` - реестр objective descriptors.
- `experiments.py` и `benchmarks.py` - запуск сравнений оптимизаторов.
- `nn.py` - класс `MLPClassifier` поверх native MLP.
- `datasets.py` - CSV, stratified split, standardization, F1, save/load.
- `studies.py` - сравнение оптимизаторов MLP, ablation, sklearn/PyTorch baseline.
- `plotting.py` - компактные функции визуализации для ноутбуков.

## Контракт результата

Оптимизаторы возвращают `dict` со стабильными ключами:

- `x` - финальная точка;
- `value` - значение функции;
- `gradient_norm` - норма градиента, если метод ее вычисляет;
- `iterations`;
- `function_evaluations`;
- `gradient_evaluations`;
- `hessian_evaluations`;
- `converged`;
- `trajectory`.

`trajectory` содержит `x`, `f`, `grad_norm`, `time_ms`. Запись траектории можно
выключить через `log_trajectory=False`, чтобы не тратить память на большие
сравнения.

## Поток данных

Типовой pipeline:

1. Python готовит `numpy.ndarray`.
2. pybind11 проверяет размерность и тип.
3. C++ копирует вход в `Vector` или `Matrix`, если нужна внутренняя ownership
   модель.
4. Оптимизатор или MLP работает без GIL.
5. Результат возвращается как NumPy массив или Python `dict`.

Такой контракт упрощает ноутбуки: графики и таблицы строятся на обычных NumPy и
pandas структурах, а тяжелые вычисления остаются в C++.
