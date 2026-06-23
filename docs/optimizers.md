# Оптимизаторы

Модуль `optimizers/FirstOrder` реализует методы первого порядка с общим
контрактом:

```text
OptimizeResult = MinimizeFirstOrder(value, gradient, x0, method, config)
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
x <- x - alpha * grad
```

HeavyBall:

```text
v <- mu * v - alpha * grad(x)
x <- x + v
```

Nesterov:

```text
grad <- grad(x + mu * v)
v <- mu * v - alpha * grad
x <- x + v
```

Adam:

```text
m <- beta1 * m + (1 - beta1) * grad
v <- beta2 * v + (1 - beta2) * grad^2
m_hat <- m / (1 - beta1^t)
v_hat <- v / (1 - beta2^t)
x <- x - alpha * m_hat / (sqrt(v_hat) + eps)
```

RMSProp:

```text
v <- beta2 * v + (1 - beta2) * grad^2
x <- x - alpha * grad / (sqrt(v) + eps)
```

Adagrad:

```text
s <- s + grad^2
x <- x - alpha * grad / (sqrt(s) + eps)
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
