# Тестовые функции

Модуль benchmark-функций нужен для проверки оптимизаторов на гладких,
мультимодальных и разрывных поверхностях. Для части функций реализованы
аналитические градиенты и гессианы, для части доступен autograd или численная
производная.

## Реестр objective

Python helper `optlib.get_objective(name, dimension=2, scale=1.0)` возвращает
объект с полями:

- `value(x)`;
- `gradient(x)` или `None`;
- `hessian(x)` или `None`;
- `minimum`;
- `minimum_value`;
- `derivative_free`;
- `native_name`.

Список имен:

```python
optlib.list_objectives()
```

## Rosenbrock

Для $n$ переменных:

$$
f(x) = \sum_{i=0}^{n-2} 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
$$

Глобальный минимум:

$$
x^* = (1, 1, \dots, 1),\quad f(x^*) = 0
$$

API:

- `RosenbrockValue(x)`;
- `RosenbrockGradient(x)`;
- `RosenbrockHessian(x)`;
- `RosenbrockNumericalGradient(x, scheme="central")`;
- `RosenbrockAutogradGradient(x)`.

## Rastrigin

$$
f(x) = 10n + \sum_{i=1}^{n} (x_i^2 - 10 \cos(2\pi x_i))
$$

Минимум:

$$
x^* = (0, \dots, 0),\quad f(x^*) = 0
$$

Функция мультимодальная и хорошо показывает риск локального застревания.

## Himmelblau

$$
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
$$

Один из минимумов:

$$
(x^*, y^*) = (3, 2),\quad f(x^*, y^*) = 0
$$

Функция двумерная, гладкая, с несколькими минимумами.

## Ackley

$$
f(x) =
-20 \exp(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2})
- \exp(\frac{1}{n} \sum_{i=1}^{n} \cos(2\pi x_i))
+ 20 + e
$$

Минимум:

$$
x^* = (0, \dots, 0),\quad f(x^*) = 0
$$

Ackley полезна для проверки поведения на почти плоских областях.

## Beale

$$
f(x, y) =
(1.5 - x + xy)^2 +
(2.25 - x + xy^2)^2 +
(2.625 - x + xy^3)^2
$$

Минимум:

$$
(x^*, y^*) = (3, 0.5),\quad f(x^*, y^*) = 0
$$

## Booth

$$
f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
$$

Минимум:

$$
(x^*, y^*) = (1, 3),\quad f(x^*, y^*) = 0
$$

## Styblinski-Tang

$$
f(x) = \frac{1}{2} \sum_{i=1}^{n} (x_i^4 - 16x_i^2 + 5x_i)
$$

Функция мультимодальная и используется для ND-проверок.

## DesmosSurface

Формула поверхности:

$$
z = d (
([x (round(\sin(10y)) + 2)]^2 + y - 10)^2
+ (x + [y (round(\sin(7x)) + 2)]^2 - 7)^2
)
$$

Параметр $d > 0$ масштабирует значения, по умолчанию `scale=1.0`.
Из-за `round(sin(...))` поверхность кусочно-гладкая и имеет разрывы, поэтому
аналитический градиент для нее не используется.

API:

```python
value = optlib.DesmosSurfaceValue(x, scale=1.0)
gradient = optlib.DesmosSurfaceNumericalGradient(x, scale=1.0, scheme="central")
```

Для оптимизации этой поверхности предпочтительны `nelder_mead`, `powell` и
`coordinate_search`.
