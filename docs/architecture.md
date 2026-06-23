# Архитектура

Библиотека разделена на три слоя: C++23-ядро, pybind11-биндинги и тонкий
Python-пакет `optlib`.

## C++ ядро

Исходники находятся в `src/optlib/core/`. Ядро не зависит от Eigen, BLAS, Boost
или CUDA: контейнеры, линейная алгебра, дифференцирование и оптимизаторы
реализованы внутри проекта.

Ключевые модули:

- `LinAlg.hpp` — плотные `Vector` и `Matrix`, `Dot`, `Norm2`, `Axpy`, `Gemv`,
  `Gemm`.
- `Dual.hpp` — forward-mode multidual числа.
- `Differentiation.hpp/.cpp` — численные градиенты и автодифференцирование.
- `functions/*.hpp/.cpp` — тестовые функции лабораторных 1-2: Rosenbrock,
  Rastrigin, Himmelblau, Ackley, DesmosSurface, Beale, Booth,
  Styblinski-Tang.
- `optimizers/FirstOrder.*` — GD, HeavyBall, Nesterov, Adam, RMSProp, Adagrad.
- `optimizers/SecondOrder.*` — Newton, BFGS, L-BFGS с line search.
- `optimizers/ZeroOrder.*` — Nelder-Mead, Powell и coordinate search для
  black-box поверхностей.
- `nn/NeuralNetwork.*` — бинарная MLP с плоским вектором параметров.
- `OptimizeResult.hpp` — результат оптимизации и отключаемая траектория.

## Биндинги

`src/optlib/bindings/Module.cpp` экспортирует расширение `_optlib`. На входе
используются C-contiguous `numpy.ndarray` с `forcecast` к `double`. Для
результатов C++ создает собственное хранилище и передает его в NumPy через
`py::capsule`, поэтому лишняя копия при возврате не нужна.

Горячие C++ участки освобождают GIL. Если целевая функция или градиент приходят
из Python, GIL захватывается только на время Python-вызова.

## Python API

`python/optlib/__init__.py` реэкспортирует публичные функции из `_optlib`.
Чистый Python-слой содержит только экспериментальную инфраструктуру:
реестр функций, графики, загрузку датасетов, study-функции и удобный класс
`MLPClassifier` поверх нативной MLP. Пакет собирается через scikit-build-core
и CMake, поэтому обычный импорт:

```python
import optlib
```

дает доступ к C++ реализации и воспроизводимым исследовательским оберткам.

## Воспроизводимые артефакты

Ноутбуки `notebooks/first_lab.ipynb`, `notebooks/second_lab.ipynb`,
`notebooks/third_lab.ipynb` и `notebooks/fourth_lab.ipynb` генерируются из
скриптов `_build_*_lab.py`. Это сохраняет единый стиль, повторяемые seed и
позволяет быстро пересобрать ноутбуки после изменения API.

Все лабораторные используют общий контракт результатов: `OptimizeResult`,
`trajectory`, таблицы `pandas.DataFrame` и функции из `optlib.plotting`.
Для закрытого d3 предусмотрены два режима: прямое применение сохраненной
бинарной модели, если формат совпадает с обучающим датасетом, и независимая
оценка через `studies.weighted_f1_score`, если d3 имеет другую размерность или
несколько классов.
