# Оптимизаторы

Оптимизаторы работают с общей структурой результата `OptimizeResult` и
возвращают Python `dict`. Все методы принимают стартовую точку `x0`, критерии
остановки и флаг `log_trajectory`.

Критерии остановки:

- максимум итераций;
- малая норма градиента;
- малый шаг;
- малое изменение значения функции.

## Методы первого порядка

Gradient Descent:

$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
$$

HeavyBall:

$$
v_{k+1} = \mu v_k - \alpha_k \nabla f(x_k),\quad
x_{k+1} = x_k + v_{k+1}
$$

Nesterov:

$$
g_k = \nabla f(x_k + \mu v_k)
$$

$$
v_{k+1} = \mu v_k - \alpha_k g_k,\quad
x_{k+1} = x_k + v_{k+1}
$$

Adam:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
$$

$$
x_{t+1} =
x_t - \alpha_t
\frac{m_t / (1 - \beta_1^t)}{\sqrt{s_t / (1 - \beta_2^t)} + \varepsilon}
$$

RMSProp:

$$
s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2,\quad
x_{t+1} = x_t - \alpha_t \frac{g_t}{\sqrt{s_t} + \varepsilon}
$$

Adagrad:

$$
s_t = s_{t-1} + g_t^2,\quad
x_{t+1} = x_t - \alpha_t \frac{g_t}{\sqrt{s_t} + \varepsilon}
$$

Python API:

```python
result = optlib.Minimize(
    value_function,
    gradient_function,
    x0,
    method="adam",
    learning_rate=1e-3,
    max_iter=10_000,
)
```

Поддерживаемые строки `method`: `"gradient_descent"`, `"heavy_ball"`,
`"nesterov"`, `"adam"`, `"rmsprop"`, `"adagrad"`.

## Learning-rate schedules

Step:

$$
\alpha_k = \alpha_0 \gamma^{\lfloor k / s \rfloor}
$$

Exponential:

$$
\alpha_k = \alpha_0 \exp(-\lambda k)
$$

Cosine:

$$
\alpha_k =
\alpha_{\min} +
\frac{1}{2}(\alpha_0 - \alpha_{\min})(1 + \cos(\pi k / T))
$$

Warmup:

$$
\alpha_k = \alpha_k \frac{k + 1}{w},\quad k < w
$$

Проверить расписание можно отдельно:

```python
lr = optlib.LearningRateAt(
    50,
    schedule="cosine",
    initial_learning_rate=0.1,
    total_iterations=1000,
)
```

## Line search

Armijo condition:

$$
f(x + \alpha p) \le f(x) + c_1 \alpha \nabla f(x)^\top p
$$

Strong Wolfe curvature condition:

$$
|\nabla f(x + \alpha p)^\top p| \le c_2 |\nabla f(x)^\top p|
$$

Если strong Wolfe не найден за ограниченное число попыток, реализация
возвращается к backtracking Armijo.

## Методы второго порядка

Newton:

$$
H p = -\nabla f(x),\quad x_{next} = x + \alpha p
$$

Система решается плотным Gaussian elimination с pivoting. Гессиан не
инвертируется явно. Если направление не является направлением спуска,
используется `-gradient`.

BFGS хранит приближение обратного гессиана $B \approx H^{-1}$:

$$
s = x_{next} - x,\quad y = \nabla f(x_{next}) - \nabla f(x)
$$

$$
\rho = \frac{1}{y^\top s}
$$

$$
B_{next} =
(I - \rho s y^\top) B (I - \rho y s^\top) + \rho s s^\top
$$

L-BFGS хранит последние $m$ пар $(s_i, y_i)$ и строит направление через
two-loop recursion. Память - $O(mn)$.

Python API:

```python
result = optlib.MinimizeSecondOrder(
    value_function,
    gradient_function,
    x0,
    method="lbfgs",
    line_search="strong_wolfe",
)
```

Поддерживаемые строки `method`: `"newton"`, `"bfgs"`, `"lbfgs"`.

## Методы нулевого порядка

Эти методы используют только значения функции:

- `nelder_mead` - симплекс с reflection, expansion, contraction и shrink;
- `powell` - покоординатные направления с одномерным поиском;
- `coordinate_search` - поиск по координатам с уменьшением радиуса.

Python API:

```python
result = optlib.MinimizeZeroOrder(
    value_function,
    x0,
    method="powell",
    max_iter=2000,
)
```

Методы нулевого порядка полезны для разрывных и black-box поверхностей, где
градиент неустойчив или недоступен.
