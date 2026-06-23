# Оптимизаторы

Модуль `optimizers/FirstOrder` реализует методы первого порядка с общим
контрактом:

```cpp
OptimizeResult = MinimizeFirstOrder(value, gradient, x0, method, config);
```

Результат содержит итоговую точку, значение функции, норму градиента, число
итераций, флаг сходимости и отключаемую траекторию.

## Критерии останова

`StopCriteria` содержит:

- `MaxIterations`;
- `GradientTolerance`;
- `StepTolerance`;
- `FunctionTolerance`.

Если `StepTolerance` или `FunctionTolerance` равны нулю, соответствующий
критерий отключается. Это удобно в экспериментах, где нужно сравнить методы до
одинакового порога по норме градиента.

## Методы

Gradient Descent:

```text
x_{k+1} = x_k - alpha * grad(f, x_k)
```

HeavyBall:

```text
v_{k+1} = mu * v_k - alpha * grad(f, x_k)
x_{k+1} = x_k + v_{k+1}
```

Nesterov:

```text
g_k     = grad(f, x_k + mu * v_k)
v_{k+1} = mu * v_k - alpha * g_k
x_{k+1} = x_k + v_{k+1}
```

Adam:

```text
m_t       = beta1*m_{t-1} + (1 - beta1)*g_t
v_t       = beta2*v_{t-1} + (1 - beta2)*g_t^2
m_hat_t   = m_t / (1 - beta1^t)
v_hat_t   = v_t / (1 - beta2^t)
x_{t+1}   = x_t - alpha * m_hat_t / (sqrt(v_hat_t) + eps)
```

RMSProp:

```text
v_t     = beta2*v_{t-1} + (1 - beta2)*g_t^2
x_{t+1} = x_t - alpha * g_t / (sqrt(v_t) + eps)
```

Adagrad:

```text
s_t     = s_{t-1} + g_t^2
x_{t+1} = x_t - alpha * g_t / (sqrt(s_t) + eps)
```

## Траектория

`Trajectory` хранит:

- точки `x_k`;
- значения `f_k`;
- нормы градиента;
- время от старта в миллисекундах.

Логирование можно отключить через `StoreTrajectory` / `log_trajectory`, чтобы
бенчмарки не платили за запись полного пути.

## Исследованные источники

Для Adam используется стандартная bias-correction формула из статьи Kingma и
Ba, где метод описан как вычислительно эффективный и малопамятный адаптивный
метод первого порядка:

- https://arxiv.org/abs/1412.6980

Для внешнего сравнения в первой лабораторной применяется официальный
`scipy.optimize.minimize`; в следующих этапах этот же интерфейс используется
как reference baseline:

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html

## Расписания learning rate

Первый порядок поддерживает фиксированный `LearningRate` и расписания:

Step:

```text
alpha_k = alpha_0 * gamma^floor(k / step_size)
```

Exponential:

```text
alpha_k = alpha_0 * exp(-lambda * k)
```

Cosine:

```text
alpha_k = alpha_min
          + 0.5 * (alpha_0 - alpha_min) * (1 + cos(pi*k / T))
```

Warmup применяется поверх выбранного расписания:

```text
if k < warmup_steps:
    alpha_k = alpha_k * (k + 1) / warmup_steps
```

Python API:

```python
optlib.LearningRateAt(10, "step", initial_learning_rate=0.1, gamma=0.5, step_size=5)
optlib.MinimizeRosenbrock(x0, method="adam", schedule="cosine", schedule_iterations=30000)
```

## Линейный поиск

Для методов второго порядка используется backtracking line search.

Armijo:

```text
f(x + alpha*p) <= f(x) + c1*alpha*grad(f, x)^T*p
```

Strong Wolfe дополнительно проверяет кривизну:

```text
|grad(f, x + alpha*p)^T*p| <= c2 * |grad(f, x)^T*p|
```

Если strong Wolfe не найден за ограниченное число попыток, реализация
возвращается к Armijo-уменьшению. Это делает метод устойчивее на кривых долинах
Розенброка.

При реализации использованы формулы из локального ТЗ и сверка с документацией
SciPy `line_search`, где указано, что алгоритм обеспечивает strong Wolfe
conditions и ссылается на Nocedal/Wright:

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html

## Методы второго порядка

Newton:

```text
H * p = -grad(f, x)
x_next = x + alpha * p
```

Система решается плотным pivoted Gaussian elimination; матрица не обращается.
На диагональ добавляется малое демпфирование `tau I`. Если направление не
является направлением спуска, используется `-grad`.

BFGS хранит плотное приближение обратного гессиана `B ~= H^{-1}`:

```text
s = x_next - x
y = grad(f, x_next) - grad(f, x)
rho = 1 / (y^T * s)
B_next = (I - rho*s*y^T) * B * (I - rho*y*s^T) + rho*s*s^T
```

Практически используется эквивалентная rank-two формула без явного построения
трех матриц. Если `y^T s` слишком мало, обновление пропускается.

L-BFGS хранит последние `m` пар `(s_i, y_i)` и строит направление через
two-loop recursion. Память: `O(mn)`, поэтому тест отдельно проверяет
Розенброка при `n=100`.

Python API:

```python
optlib.MinimizeRosenbrockSecondOrder(x0, method="lbfgs", history_size=12)
optlib.MinimizeSecondOrder(value, gradient, x0, method="bfgs")
optlib.MinimizeSecondOrder(value, gradient, x0, method="newton", hessian_function=hessian)
```

SciPy-документация по `minimize` использовалась как внешний reference baseline:
там перечислены `BFGS`, `Nelder-Mead`, `Powell`, `L-BFGS-B` и пример
минимизации Розенброка.

## Методы нулевого порядка

Методы нулевого порядка используют только значения функции. Это важно для
разрывных или black-box поверхностей следующих лабораторных.

Nelder-Mead хранит симплекс из `n + 1` вершин:

- reflection `alpha = 1`;
- expansion `gamma = 2`;
- contraction `rho = 0.5`;
- shrink `sigma = 0.5`.

Остановка проверяет размер симплекса и разброс значений, а не только движение
лучшей точки. Это предотвращает раннюю остановку, когда лучший vertex временно
не меняется.

Powell и coordinate search используют одномерную минимизацию golden section
вдоль направлений. Powell после цикла добавляет направление суммарного
смещения, а coordinate search циклически проходит по базисным координатам.

Python API:

```python
optlib.MinimizeZeroOrder(value, x0, method="nelder_mead")
optlib.MinimizeRosenbrockZeroOrder(x0, method="powell")
optlib.MinimizeRosenbrockZeroOrder(x0, method="coordinate_search")
```

Документация SciPy по Nelder-Mead и Powell использовалась как сверка
терминологии и ожидаемого назначения методов:

- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html

## Сравнительный запуск в лабораторной 2

Для экспериментов методы запускаются через общий Python-харнесс:

```python
objective = optlib.get_objective("ackley", dimension=10)
rows = optlib.compare_methods(
    objective,
    x0,
    ["adam", "rmsprop", "lbfgs", "nelder_mead", "powell"],
    max_iter=1000,
    log_trajectory=False,
)
```

Для гладких функций используются методы первого и второго порядка. Для
`DesmosSurface`, где поверхность разрывна из-за `round(sin(...))`, основное
сравнение ведётся по методам нулевого порядка. Численный градиент этой
поверхности используется только как диагностический локальный инструмент, а не
как математическая гарантия гладкости.
