# Бенчмарки

Этот раздел фиксирует способ воспроизведения измерений. Полные сравнительные
таблицы будут расширяться в следующих лабораторных, когда появятся методы
нулевого порядка, методы второго порядка и нейросетевая часть.

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

## Ограничения измерений

`OptimizeResult` сейчас хранит число итераций, но не число вызовов функции и
градиента. Поэтому для методов первого порядка в первой лабораторной честно
сравниваются итоговое значение, норма градиента, число итераций и wall-clock
время, а не `nfev`/`njev`.
