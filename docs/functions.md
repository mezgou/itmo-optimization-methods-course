# Тестовые функции

Первый набор экспериментов построен вокруг функции Розенброка. Она гладкая,
имеет известный минимум и узкую изогнутую долину, поэтому хорошо показывает
различия между методами первого порядка.

## Rosenbrock

Для `n` переменных:

```text
f(x) = sum_{i=0}^{n-2} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
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
