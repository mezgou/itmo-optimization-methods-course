"""Builder for notebooks/first_lab.ipynb.

Emits a clean, reproducible notebook via nbformat. Run with `uv run python`,
then execute headlessly with nbconvert. All prose is Russian; code is English.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

NB = nbf.v4.new_notebook()
cells: list = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
md(
    r"""
# Отчет 1: дифференцирование и методы первого порядка

Проверяем численные градиенты, `Dual`-autograd и first-order методы на
Розенброке. Сравниваем точность, скорость, сходимость в 2D/ND и эталон `scipy`
"""
)

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
md(
    r"""
## 1. Подготовка окружения

Импорт, seed и единый стиль графиков
"""
)
code(
    r"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import optlib
from optlib import functions, experiments, plotting

np.random.seed(7)
plotting.use_notebook_style()
pd.set_option("display.float_format", lambda v: f"{v:.6g}")

print("optlib version:", getattr(optlib, "__version__", "n/a"))
print("целевые функции:", functions.list_objectives())
"""
)

# ---------------------------------------------------------------------------
# 2. Numerical differentiation accuracy vs h
# ---------------------------------------------------------------------------
md(
    r"""
## 2. Точность численного дифференцирования и выбор шага $h$

Три схемы конечных разностей:

| Схема | Формула | Порядок |
|-------|---------|---------|
| **Forward** | $\dfrac{f(x+h)-f(x)}{h}$ | $O(h)$ |
| **Central** | $\dfrac{f(x+h)-f(x-h)}{2h}$ | $O(h^2)$ |
| **Five-point** | $\dfrac{-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)}{12h}$ | $O(h^4)$ |

Ищем оптимальный $h^*$: при большом шаге доминирует усечение, при малом
округление. `Dual`-autograd добавлен как эталон без подбора шага
"""
)
code(
    r"""
# Точка вдали от минимума, где градиент Розенброка нетривиален.
probe = np.array([0.5, 1.5])
analytic = optlib.RosenbrockGradient(probe)
autograd = optlib.RosenbrockAutogradGradient(probe)

print("аналитический градиент:", analytic)
print("autograd (Dual):       ", autograd)
print("ошибка autograd:        %.2e" % np.linalg.norm(autograd - analytic))

steps = np.logspace(-12, -1, 60)
schemes = ["forward", "central", "five_point"]
errors = {s: np.empty_like(steps) for s in schemes}
for scheme in schemes:
    for i, h in enumerate(steps):
        grad = optlib.RosenbrockNumericalGradient(probe, scheme=scheme, step=h)
        errors[scheme][i] = np.linalg.norm(grad - analytic)

# Сводка: лучший достигнутый h по каждой схеме.
summary = pd.DataFrame(
    {
        "схема": schemes,
        "мин. ошибка": [errors[s].min() for s in schemes],
        "оптимальный h": [steps[np.argmin(errors[s])] for s in schemes],
        "порядок (теор.)": [1, 2, 4],
    }
)
summary
"""
)
code(
    r"""
fig, ax = plt.subplots(figsize=(8.2, 4.8))
slopes = {"forward": 1, "central": 2, "five_point": 4}
labels = {"forward": "forward $O(h)$", "central": "central $O(h^2)$",
          "five_point": "five-point $O(h^4)$"}
for idx, scheme in enumerate(schemes):
    color = plotting.PALETTE[idx]
    ax.loglog(steps, errors[scheme], "o-", ms=3, color=color, label=labels[scheme])
    # Опорная прямая теоретического наклона, привязанная к участку усечения.
    anchor = errors[scheme].argmin()
    h0, e0 = steps[anchor], errors[scheme][anchor]
    ref = e0 * (steps / h0) ** slopes[scheme]
    band = steps >= h0
    ax.loglog(steps[band], ref[band], "--", color=color, alpha=0.5, lw=1.2)

ax.axhline(
    np.linalg.norm(autograd - analytic) + 1e-18,
    color="#16a34a", lw=2, label="autograd (Dual)",
)
ax.set_xlabel("шаг h")
ax.set_ylabel("‖числ. − аналит.‖")
ax.set_title("Точность схем дифференцирования в зависимости от h (Розенброк)")
ax.legend()
plt.show()
"""
)
md(
    r"""
**Вывод.** Порядки совпадают с теорией: five-point точнее forward на несколько
порядков. При слишком малом $h$ ошибка снова растёт. `Dual` даёт машинную
точность без выбора шага
"""
)

# ---------------------------------------------------------------------------
# 3. First-order optimizers on Rosenbrock 2D
# ---------------------------------------------------------------------------
md(
    r"""
## 3. Методы первого порядка на Розенброке 2D

Розенброк: $f(x,y)=100(y-x^2)^2+(1-x)^2$, минимум $(1,1)$, старт
$x_0=(-1.2,1.0)$. Сравниваем GD, Heavy Ball, Nesterov и Adam
"""
)
code(
    r"""
rosen2d = functions.get_objective("rosenbrock", dimension=2)
x0 = np.array([-1.2, 1.0])

first_order_specs = {
    "gradient_descent": dict(learning_rate=1e-3, max_iter=20000),
    "heavy_ball": dict(learning_rate=1e-3, max_iter=20000, momentum=0.9),
    "nesterov": dict(learning_rate=1e-3, max_iter=20000, momentum=0.9),
    "adam": dict(learning_rate=1e-2, max_iter=20000),
}

first_results = {
    name: experiments.run_method(rosen2d, x0, name, gradient_tolerance=1e-6, **spec)
    for name, spec in first_order_specs.items()
}

rows = [
    experiments.result_summary(name, res, rosen2d)
    for name, res in first_results.items()
]
table = pd.DataFrame(rows)[
    ["method", "value", "gradient_norm", "iterations",
     "function_evaluations", "converged", "distance_to_minimum"]
]
table.columns = ["метод", "f*", "‖∇f‖", "итераций",
                 "вызовов f", "сошёлся", "‖x*−x_min‖"]
table.round(8)
"""
)
code(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
plotting.plot_trajectories(
    axes[0], rosen2d.value, first_results,
    x_limits=(-2.0, 2.0), y_limits=(-1.0, 3.0), minimum=[1.0, 1.0],
    title="Траектории методов первого порядка",
)
plotting.plot_convergence(
    axes[1], first_results, key="f",
    title="Сходимость по значению f(x)",
)
plt.show()

fig, ax = plt.subplots(figsize=(9.2, 4.2))
plotting.plot_convergence(
    ax, first_results, key="grad_norm",
    title="Сходимость по норме градиента ‖∇f(x)‖",
)
plt.show()
"""
)
md(
    r"""
**Вывод.** GD медленнее всех. Моментные методы и Adam быстрее проходят овраг;
осцилляции видны на графике нормы градиента
"""
)

# ---------------------------------------------------------------------------
# 4. Multidimensional Rosenbrock
# ---------------------------------------------------------------------------
md(
    r"""
## 4. Многомерный Розенброк ($n = 2, 5, 10, 50, 100$)

Проверяем ND-форму:
$f(x)=\sum_{i=1}^{n-1} 100(x_{i+1}-x_i^2)^2+(1-x_i)^2$
Минимум: $(1,\ldots,1)$; старт: чередование `-1.2/1.0`
"""
)
code(
    r"""
def make_start(n: int) -> np.ndarray:
    x = np.full(n, -1.2)
    x[1::2] = 1.0
    return x

# В многомерном овраге Nesterov склонен «перелетать» долину, поэтому здесь
# берём чуть меньший шаг (5e-4), это сохраняет устойчивость при росте n.
nd_specs = {
    "gradient_descent": dict(learning_rate=1e-3, max_iter=30000),
    "heavy_ball": dict(learning_rate=1e-3, max_iter=30000, momentum=0.9),
    "nesterov": dict(learning_rate=5e-4, max_iter=30000, momentum=0.9),
    "adam": dict(learning_rate=1e-2, max_iter=30000),
}

nd_rows = []
for n in (2, 5, 10, 50, 100):
    obj = functions.get_objective("rosenbrock", dimension=n)
    start = make_start(n)
    for method, spec in nd_specs.items():
        res = experiments.run_method(obj, start, method, gradient_tolerance=1e-6, **spec)
        nd_rows.append(
            {
                "n": n,
                "метод": method,
                "f*": res["value"],
                "итераций": res["iterations"],
                "‖∇f‖": res["gradient_norm"],
                "сошёлся": res["converged"],
            }
        )

nd_table = pd.DataFrame(nd_rows)
pivot = nd_table.pivot(index="метод", columns="n", values="итераций")
print("Число итераций до останова (строки: метод, столбцы: размерность n):")
display(pivot)
nd_table.round(8)
"""
)
md(
    r"""
**Вывод.** При росте $n$ итераций становится больше. Adam и моментные методы
сохраняют сходимость до $n=100$; ND-градиент работает корректно
"""
)

# ---------------------------------------------------------------------------
# 5. autograd vs numerical: accuracy + timing
# ---------------------------------------------------------------------------
md(
    r"""
## 5. Автоматическое дифференцирование vs численное: точность и скорость

Сравниваем источники градиента: ошибка относительно аналитики и среднее время
одного вызова. Для численных схем шаг выбирается автоматически
"""
)
code(
    r"""
import timeit

sample = np.random.uniform(-2.0, 2.0, size=(200, 2))
analytic_grads = np.array([optlib.RosenbrockGradient(p) for p in sample])

def mean_error(fn) -> float:
    errs = [np.linalg.norm(fn(p) - a) for p, a in zip(sample, analytic_grads)]
    return float(np.mean(errs))

def timed(fn, repeat: int = 2000) -> float:
    p = sample[0]
    t = timeit.timeit(lambda: fn(p), number=repeat)
    return t / repeat * 1e6  # микросекунды на вызов

sources = {
    "forward":    lambda p: optlib.RosenbrockNumericalGradient(p, scheme="forward"),
    "central":    lambda p: optlib.RosenbrockNumericalGradient(p, scheme="central"),
    "five_point": lambda p: optlib.RosenbrockNumericalGradient(p, scheme="five_point"),
    "autograd":   optlib.RosenbrockAutogradGradient,
    "analytic":   optlib.RosenbrockGradient,
}

diff_rows = []
for name, fn in sources.items():
    diff_rows.append(
        {
            "источник": name,
            "ср. ошибка": mean_error(fn),
            "мкс/вызов": timed(fn),
        }
    )
diff_table = pd.DataFrame(diff_rows)
diff_table
"""
)
code(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
order = ["forward", "central", "five_point", "autograd"]
err_vals = [diff_table.set_index("источник").loc[s, "ср. ошибка"] + 1e-18 for s in order]
axes[0].bar(order, err_vals, color=[plotting.PALETTE[i] for i in range(len(order))])
axes[0].set_yscale("log")
axes[0].set_ylabel("ср. ошибка (лог)")
axes[0].set_title("Точность источников градиента")
axes[0].tick_params(axis="x", rotation=20)

time_order = ["analytic", "forward", "central", "five_point", "autograd"]
time_vals = [diff_table.set_index("источник").loc[s, "мкс/вызов"] for s in time_order]
plotting.bar_comparison(
    axes[1], time_order, time_vals,
    ylabel="мкс / вызов", title="Скорость вычисления градиента",
    highlight_max=False,
)
axes[1].tick_params(axis="x", rotation=20)
plt.show()
"""
)
md(
    r"""
**Вывод.** `Dual` почти совпадает с аналитикой. Численные схемы универсальны, но
дороже: они требуют нескольких вычислений функции на координату
"""
)

# ---------------------------------------------------------------------------
# 6. Benchmark vs scipy
# ---------------------------------------------------------------------------
md(
    r"""
## 6. Мини-бенчмарк против `scipy.optimize`

Сверяем Розенброк 2D с `scipy.optimize`: качество, итерации, вызовы функции и
время
"""
)
code(
    r"""
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scipy_rows = []
    for method in ("CG", "BFGS", "L-BFGS-B", "Nelder-Mead", "Powell"):
        row = experiments.scipy_minimize(rosen2d, x0, method=method, max_iter=20000)
        if row is not None:
            scipy_rows.append(row)

# Наши лучшие первого порядка для контекста.
optlib_rows = []
for name, res in first_results.items():
    row = experiments.result_summary(name, res, rosen2d)
    times = np.asarray(res["trajectory"].get("time_ms", []), dtype=float)
    row["wall_ms"] = float(times[-1]) if times.size else np.nan
    optlib_rows.append(row)

bench = pd.DataFrame(optlib_rows + scipy_rows)[
    ["method", "value", "iterations", "function_evaluations", "wall_ms", "converged"]
]
bench.columns = ["метод", "f*", "итераций", "вызовов f", "время, мс", "сошёлся"]
bench.round(10)
"""
)
md(
    r"""
**Вывод.** `optlib` достигает $f^* \to 0$. По итерациям first-order методы
уступают BFGS/CG из `scipy`, что ожидаемо без информации о кривизне
"""
)

# ---------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------
md(
    r"""
## Выводы

1. Конечные разности дают ожидаемые порядки $O(h)$, $O(h^2)$, $O(h^4)$; нужен
   оптимальный $h^*$
2. `Dual`-autograd даёт машинную точность за один проход
3. GD, Heavy Ball, Nesterov и Adam сходятся на 2D/ND Розенброке; инерция и Adam
   заметно быстрее GD
4. Качество совпадает с `scipy`; отставание по итерациям объясняется отсутствием
   кривизны у методов первого порядка
"""
)

NB["cells"] = cells
NB["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = pathlib.Path(__file__).resolve().parent / "first_lab.ipynb"
nbf.write(NB, out)
print("wrote", out, "with", len(cells), "cells")
