# Тестовые функции

Первый набор экспериментов построен вокруг функции Розенброка. Она гладкая,
имеет известный минимум и узкую изогнутую долину, поэтому хорошо показывает
различия между методами первого порядка.

## Rosenbrock

Для `n` переменных:

```text
f(x) = sum_{i=0}^{n-2} [100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
```

Глобальный минимум достигается в точке:

```text
x* = (1, 1, ..., 1), f(x*) = 0
```

В `optlib` реализованы:

- `RosenbrockValue(x)`;
- `RosenbrockGradient(x)`;
- `RosenbrockHessian(x)`;
- `RosenbrockNumericalGradient(x, scheme, step)`;
- `RosenbrockAutogradGradient(x)`.

## Проверки

Аналитический градиент сверяется с:

- центральной и пятиточечной конечной разностью;
- multidual autograd;
- тестами с размерностями 2D и ND.

## Эталон SciPy

В ноутбуке используется `scipy.optimize.minimize` как внешний эталон для
Rosenbrock. По официальному руководству SciPy функция `minimize` дает единый
интерфейс к многомерной минимизации, а документация демонстрирует ее именно на
Rosenbrock:

- https://docs.scipy.org/doc/scipy/tutorial/optimize.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

SciPy не является зависимостью C++ ядра. Он подключается только для
экспериментальных сравнений через `uv sync --extra experiments`.

## Rastrigin

Многомерная функция Растригина выбрана как обязательная функция из списка
классических тестовых задач:

```text
f(x) = 10*n + sum_{i=1}^{n} [x_i^2 - 10*cos(2*pi*x_i)]
```

Глобальный минимум:

```text
x* = (0, ..., 0), f(x*) = 0
```

Поверхность сильно мультимодальная. Для локальных методов это стресс-тест на
застревание в ближайшей яме, поэтому в исследовании важны multistart и доля
успешных запусков, а не только один красивый старт.

В `optlib` реализованы:

- `RastriginValue(x)`;
- `RastriginGradient(x)`;
- `RastriginHessian(x)`;
- `RastriginAutogradGradient(x)`.

## Himmelblau

Двумерная функция Химмельблау используется как гладкая поверхность с несколькими
минимумами:

```text
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
```

Один из минимумов:

```text
(x*, y*) = (3, 2), f(x*, y*) = 0
```

Реализованы аналитические значение, градиент, гессиан и autograd-проверка:

- `HimmelblauValue(x)`;
- `HimmelblauGradient(x)`;
- `HimmelblauHessian(x)`;
- `HimmelblauAutogradGradient(x)`.

## Ackley

Ackley добавлена как гладкая ND-функция с почти плоскими областями вдали от
минимума:

```text
f(x) = -20*exp(-0.2*sqrt((1/n) * sum_{i=1}^{n} x_i^2))
       - exp((1/n) * sum_{i=1}^{n} cos(2*pi*x_i))
       + 20 + e
```

Глобальный минимум:

```text
x* = (0, ..., 0), f(x*) = 0
```

Для неё доступны `AckleyValue(x)`, `AckleyGradient(x)` и
`AckleyAutogradGradient(x)`. Гессиан намеренно не вынесен в публичный API:
для лабораторного сравнения Newton/BFGS/L-BFGS достаточно аналитического
градиента, а `BFGS`/`L-BFGS` строят кривизну по траектории.

## DesmosSurface

Функция из Desmos 3D берётся строго из контракта:

```text
z = d * (
      ([x * (round(sin(10*y)) + 2)]^2 + y - 10)^2
    + (x + [y * (round(sin(7*x)) + 2)]^2 - 7)^2
)
```

`d > 0` масштабирует значения и по умолчанию равен `1.0`. Из-за
`round(sin(t))` поверхность кусочно-гладкая и имеет разрывы на границах
полос. Поэтому она рассматривается как black-box задача: основной честный
инструмент: Nelder-Mead, Powell и coordinate search. Численный градиент
доступен только для локальной диагностики внутри гладкого участка:

- `DesmosSurfaceValue(x, scale=1.0)`;
- `DesmosSurfaceNumericalGradient(x, scale=1.0, scheme="central")`.

Во второй лабораторной эта функция анализируется отдельно: строятся heatmap,
срезы вдоль осей, траектории zero-order методов и диагностика скачков
численного градиента. Это важно, потому что на границах полос обычные
first-order предположения о гладкости нарушаются.

## Дополнительные функции

Для расширенного исследования также доступны:

- Beale: 2D, минимум `(3, 0.5)`;
- Booth: 2D, минимум `(1, 3)`;
- Styblinski-Tang: ND, минимум около `-2.903534` по каждой координате.

Для них реализованы value/gradient/hessian и имена в Python-реестре:
`"beale"`, `"booth"`, `"styblinski_tang"`.

## Objective registry

Для ноутбуков и сравнительных таблиц добавлен Python-реестр:

```python
objective = optlib.get_objective("rastrigin", dimension=10)
rows = optlib.compare_methods(objective, x0, ["adam", "lbfgs", "nelder_mead"])
```

`Objective` хранит `value`, `gradient`, `hessian`, известный минимум и флаг
`derivative_free`. Это один контракт для функций лабораторий 1-2.

Во второй лабораторной через этот контракт строится матрица
`method × function × dimension`: Adam, L-BFGS и Nelder-Mead сравниваются на
Rosenbrock, Rastrigin и Ackley для размерностей 2, 10 и 50. Отдельный блок
чувствительности показывает влияние шага градиентного спуска и схемы
численного градиента (`forward`, `central`, `five_point`, `autograd`).
