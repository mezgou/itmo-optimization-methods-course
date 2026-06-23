# Численное дифференцирование

Модуль `Differentiation` вычисляет градиент функции
$f: \mathbb{R}^n \to \mathbb{R}$ конечными разностями. Каждая координата
возмущается отдельно, поэтому метод подходит для black-box функций, где можно
получить только значение функции.

## Схемы

Односторонняя схема:

$$
\frac{\partial f}{\partial x_i}(x) \approx
\frac{f(x + h e_i) - f(x)}{h}
$$

Порядок ошибки - $O(h)$.

Центральная схема:

$$
\frac{\partial f}{\partial x_i}(x) \approx
\frac{f(x + h e_i) - f(x - h e_i)}{2h}
$$

Порядок ошибки - $O(h^2)$.

Пятиточечная схема:

$$
\frac{\partial f}{\partial x_i}(x) \approx
\frac{-f(x + 2h e_i) + 8 f(x + h e_i) - 8 f(x - h e_i) + f(x - 2h e_i)}{12h}
$$

Порядок ошибки - $O(h^4)$.

## Выбор шага

Если `step=0.0`, библиотека выбирает адаптивный шаг:

$$
h_i = b \max(1, |x_i|)
$$

Значение $b$ зависит от схемы:

- forward: $b \approx \sqrt{\epsilon}$;
- central: $b \approx \epsilon^{1/3}$;
- five-point: $b \approx \epsilon^{1/5}$.

Здесь $\epsilon$ - машинная точность `double`. Такой выбор балансирует ошибку
усечения и ошибку округления.

## Python API

```python
import numpy as np
import optlib

def f(x: np.ndarray) -> float:
    return float(np.sum(x * x))

x = np.array([1.0, -2.0, 0.5])
g = optlib.NumericGradient(f, x, scheme="central", step=0.0)
```

Поддерживаемые строки: `"forward"`, `"central"`, `"five_point"`.

Для функции Розенброка есть быстрый wrapper:

```python
g = optlib.RosenbrockNumericalGradient(x, scheme="five_point")
```

## Ограничения

Конечные разности чувствительны к шуму, разрывам и слишком малому шагу. Для
гладких функций их удобно использовать как независимую проверку аналитического
градиента. Для разрывных поверхностей лучше применять методы нулевого порядка.
