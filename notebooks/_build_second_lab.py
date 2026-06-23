"""Builder for notebooks/second_lab.ipynb.

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
# Лабораторная работа №2 — Сравнительное исследование методов оптимизации

Цель: сравнить методы первого, второго и нулевого порядка на гладких,
мультимодальных и разрывных функциях. Отдельно проверяем масштабирование,
multistart, DesmosSurface и `scipy`.
"""
)

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
md(
    r"""
## 1. Подготовка окружения
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

FIRST = ["gradient_descent", "heavy_ball", "nesterov", "adam", "rmsprop", "adagrad"]
SECOND = ["newton", "bfgs", "lbfgs"]
ZERO = ["nelder_mead", "powell", "coordinate_search"]
print("методы:", FIRST, "|", SECOND, "|", ZERO)
print("функции:", functions.list_objectives())
"""
)

# ---------------------------------------------------------------------------
# 2. Objective gallery
# ---------------------------------------------------------------------------
md(
    r"""
## 2. Галерея тестовых функций

Пять типов ландшафта:

- **Розенброк** — гладкий, но узкий изогнутый овраг (минимум $(1,1)$);
- **Растригин** — сильно мультимодальный, десятки локальных минимумов
  (глобальный — $(0,0)$);
- **Химмельблау** — четыре равных по высоте минимума;
- **Экли** — почти плоское «поле» с глубокой воронкой в центре;
- **DesmosSurface** — модулированный Химмельблау с **разрывами** на границах
  полос (black-box-кейс для методов нулевого порядка).
"""
)
code(
    r"""
gallery = [
    ("rosenbrock", (-2.0, 2.0), (-1.0, 3.0), "Розенброк"),
    ("rastrigin", (-5.12, 5.12), (-5.12, 5.12), "Растригин"),
    ("himmelblau", (-5.0, 5.0), (-5.0, 5.0), "Химмельблау"),
    ("ackley", (-5.0, 5.0), (-5.0, 5.0), "Экли"),
    ("desmos_surface", (-5.0, 5.0), (-5.0, 5.0), "DesmosSurface"),
]

for name, xlim, ylim, title in gallery:
    obj = functions.get_objective(name, dimension=2)
    fig = plt.figure(figsize=(10.4, 4.0))
    plotting.plot_surface3d(
        fig, obj.value, x_limits=xlim, y_limits=ylim,
        points=110, title=f"{title} — поверхность", position=121,
    )
    ax2 = fig.add_subplot(122)
    plotting.plot_contours(
        ax2, obj.value, x_limits=xlim, y_limits=ylim,
        points=150, title=f"{title} — контуры",
        log_levels=(name in {"rosenbrock", "desmos_surface"}),
    )
    plt.show()
"""
)
md(
    r"""
**Вывод.** Розенброк проверяет кривизну, Растригин — локальные минимумы,
Ackley — масштаб шага, DesmosSurface — отказ от градиента.
"""
)

# ---------------------------------------------------------------------------
# 3. Cross-class comparison on rosenbrock & rastrigin
# ---------------------------------------------------------------------------
md(
    r"""
## 3. Сравнение классов методов на Розенброке и Растригине (2D)

Сравниваем классы методов по траектории, сходимости и таблице: $f^*$, итерации,
вызовы функции и время.
"""
)
code(
    r"""
def run_subset(obj, x0, specs):
    # specs: dict вида {метод: kwargs}; возвращает {метод: результат}.
    return {m: experiments.run_method(obj, x0, m, **kw) for m, kw in specs.items()}

# Розенброк: по одному яркому представителю каждого класса.
rosen = functions.get_objective("rosenbrock", dimension=2)
r_x0 = np.array([-1.2, 1.0])
rosen_specs = {
    "nesterov": dict(learning_rate=1e-3, max_iter=20000, momentum=0.9),
    "adam": dict(learning_rate=1e-2, max_iter=20000),
    "bfgs": dict(max_iter=2000),
    "lbfgs": dict(max_iter=2000),
    "nelder_mead": dict(max_iter=2000),
}
rosen_results = run_subset(rosen, r_x0, rosen_specs)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
plotting.plot_trajectories(
    axes[0], rosen.value, rosen_results,
    x_limits=(-2.0, 2.0), y_limits=(-1.0, 3.0), minimum=[1.0, 1.0],
    title="Розенброк: траектории по классам методов",
)
plotting.plot_convergence(axes[1], rosen_results, key="f",
                          title="Розенброк: сходимость f(x)")
plt.show()
"""
)
code(
    r"""
# Полная сводная таблица по всем методам всех классов (Розенброк).
all_methods = FIRST + SECOND + ZERO
rosen_rows = experiments.compare_methods(rosen, r_x0, all_methods, max_iter=20000)
rosen_table = pd.DataFrame(rosen_rows)[
    ["method", "value", "iterations", "function_evaluations",
     "gradient_evaluations", "wall_ms", "converged", "distance_to_minimum"]
]
rosen_table.columns = ["метод", "f*", "итер.", "выз. f", "выз. ∇f",
                       "время, мс", "сошёлся", "‖x*−x_min‖"]
rosen_table.round(6)
"""
)
code(
    r"""
# Растригин: мультимодальный. Старт рядом, но не в глобальном минимуме.
rastrigin = functions.get_objective("rastrigin", dimension=2)
ra_x0 = np.array([2.0, -1.5])
rastrigin_specs = {
    "adam": dict(learning_rate=1e-2, max_iter=3000),
    "bfgs": dict(max_iter=2000),
    "nelder_mead": dict(max_iter=2000),
    "powell": dict(max_iter=2000),
}
rastrigin_results = run_subset(rastrigin, ra_x0, rastrigin_specs)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
plotting.plot_trajectories(
    axes[0], rastrigin.value, rastrigin_results,
    x_limits=(-5.12, 5.12), y_limits=(-5.12, 5.12), minimum=[0.0, 0.0],
    title="Растригин: траектории по классам методов",
)
plotting.plot_convergence(axes[1], rastrigin_results, key="f",
                          title="Растригин: сходимость f(x)")
plt.show()

rastrigin_rows = experiments.compare_methods(rastrigin, ra_x0, all_methods, max_iter=3000)
rastrigin_table = pd.DataFrame(rastrigin_rows)[
    ["method", "value", "iterations", "wall_ms", "converged", "distance_to_minimum"]
]
rastrigin_table.columns = ["метод", "f*", "итер.", "время, мс", "сошёлся", "‖x*−x_min‖"]
rastrigin_table.round(6)
"""
)
md(
    r"""
**Вывод.** На Розенброке BFGS/L-BFGS резко сокращают число итераций. На
Растригине один старт не гарантирует глобальный минимум — нужен multistart.
"""
)

md(
    r"""
### 3.1. Матрица `method × function × dimension`

Компактный срез: `adam`, `lbfgs`, `nelder_mead` × Rosenbrock/Rastrigin/Ackley ×
размерности `2/10/50`.
"""
)
code(
    r"""
def start_for(function_name: str, n: int) -> np.ndarray:
    if function_name == "rosenbrock":
        x = np.full(n, -1.2)
        x[1::2] = 1.0
        return x
    if function_name == "rastrigin":
        return np.linspace(2.0, -1.5, n)
    return np.full(n, 2.5)

matrix_methods = {
    "adam": dict(learning_rate=1e-2, max_iter=2500),
    "lbfgs": dict(max_iter=1000),
    "nelder_mead": dict(max_iter=600),
}
matrix_functions = ("rosenbrock", "rastrigin", "ackley")
matrix_dims = (2, 10, 50)

matrix_rows = []
for function_name in matrix_functions:
    for n in matrix_dims:
        obj = functions.get_objective(function_name, dimension=n)
        start = start_for(function_name, n)
        for method, kwargs in matrix_methods.items():
            try:
                res = experiments.run_method(
                    obj, start, method, gradient_tolerance=1e-5,
                    log_trajectory=False, **kwargs,
                )
                matrix_rows.append(
                    {
                        "function": function_name,
                        "n": n,
                        "method": method,
                        "f*": res["value"],
                        "iterations": res["iterations"],
                        "calls_f": res.get("function_evaluations", 0),
                        "converged": res["converged"],
                    }
                )
            except ValueError as exc:
                matrix_rows.append(
                    {
                        "function": function_name, "n": n, "method": method,
                        "f*": np.nan, "iterations": np.nan, "calls_f": np.nan,
                        "converged": False, "note": str(exc),
                    }
                )

matrix = pd.DataFrame(matrix_rows)
display(
    matrix.pivot_table(index=["function", "n"], columns="method", values="f*")
    .style.format("{:.4g}")
    .background_gradient(cmap="viridis_r", axis=None)
)
matrix[["function", "n", "method", "f*", "iterations", "calls_f", "converged"]].round(6)
"""
)
md(
    r"""
**Вывод.** L-BFGS лучше масштабируется на гладких задачах. Nelder–Mead полезен
как black-box baseline, но быстро дорожает с ростом размерности.
"""
)

# ---------------------------------------------------------------------------
# 4. DesmosSurface case study
# ---------------------------------------------------------------------------
md(
    r"""
## 4. Кейс DesmosSurface: разрывная функция и методы нулевого порядка

Формула из контракта:

$$
z=d\left(\left([x(\operatorname{round}(\sin(10y))+2)]^2+y-10\right)^2
+\left(x+[y(\operatorname{round}(\sin(7x))+2)]^2-7\right)^2\right).
$$

Из-за `round(sin(...))` поверхность разрывна на границах полос. Градиентные
методы отклоняются; честный выбор — Nelder-Mead, Powell, coordinate search.
"""
)
code(
    r"""
desmos = functions.get_objective("desmos_surface", dimension=2)

# (а) Градиентный метод на разрывной функции -> отклоняется.
try:
    experiments.run_method(desmos, np.array([3.0, 2.0]), "adam",
                           learning_rate=1e-3, max_iter=100)
except ValueError as exc:
    print("Запуск adam на DesmosSurface отклонён:")
    print("   ", exc)

# (б) Шум численного градиента вдоль среза y = 2: значение растёт гладко,
#     а численный градиент скачет на границах полос.
xs = np.linspace(-4.0, 4.0, 33)
vals = np.array([desmos.value(np.array([x, 2.0])) for x in xs])
grad_x = np.array([
    functions.desmos_numerical_gradient(np.array([x, 2.0]), scheme="central")[0]
    for x in xs
])

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
axes[0].plot(xs, vals, color=plotting.PALETTE[0])
axes[0].set_title("DesmosSurface: значение вдоль y = 2")
axes[0].set_xlabel("x")
axes[0].set_ylabel("z")
axes[1].plot(xs, grad_x, color=plotting.PALETTE[1], marker=".")
axes[1].set_title("Численный градиент ∂z/∂x — скачки на границах полос")
axes[1].set_xlabel("x")
axes[1].set_ylabel("∂z/∂x (central)")
plt.show()
"""
)
code(
    r"""
# (в) Методы нулевого порядка на DesmosSurface из x0 = (3, 2).
d_x0 = np.array([3.0, 2.0])
desmos_results = {m: experiments.run_method(desmos, d_x0, m, max_iter=3000) for m in ZERO}

fig, ax = plt.subplots(figsize=(7.4, 5.6))
plotting.plot_contours(
    ax, desmos.value, x_limits=(-5.0, 5.0), y_limits=(-5.0, 5.0),
    points=150, title="DesmosSurface: траектории методов 0-го порядка",
    log_levels=True,
)
for idx, (name, res) in enumerate(desmos_results.items()):
    path = np.asarray(res["trajectory"]["x"])
    ax.plot(path[:, 0], path[:, 1], lw=1.6, color=plotting.PALETTE[idx], label=name)
    ax.plot(path[-1, 0], path[-1, 1], "*", ms=13, color=plotting.PALETTE[idx], mec="black")
ax.legend(loc="upper right", fontsize=9)
plt.show()

desmos_rows = [experiments.result_summary(m, r, desmos) for m, r in desmos_results.items()]
desmos_table = pd.DataFrame(desmos_rows)[
    ["method", "value", "iterations", "function_evaluations", "converged"]
]
desmos_table.columns = ["метод", "f*", "итераций", "вызовов f", "сошёлся"]
desmos_table.round(4)
"""
)
md(
    r"""
**Вывод.** Численный градиент скачет на границах полос. Nelder-Mead из
$x_0=(3,2)$ даёт лучший результат ($f^*\approx0.6$); остальные 0-order методы
могут застревать в соседних полосах.
"""
)

# ---------------------------------------------------------------------------
# 5. Multidimensional scaling
# ---------------------------------------------------------------------------
md(
    r"""
## 5. Масштабирование по размерности ($n = 2, 10, 50, 100$)

Проверяем Rosenbrock ND до $n=100$. Главное сравнение: BFGS, L-BFGS и Adam.
"""
)
code(
    r"""
def make_start(n: int) -> np.ndarray:
    x = np.full(n, -1.2)
    x[1::2] = 1.0
    return x

scale_specs = {
    "lbfgs": dict(max_iter=5000),
    "bfgs": dict(max_iter=5000),
    "adam": dict(learning_rate=1e-3, max_iter=20000),
}
dims = [2, 10, 50, 100]

scale_rows = []
for n in dims:
    obj = functions.get_objective("rosenbrock", dimension=n)
    start = make_start(n)
    for method, kw in scale_specs.items():
        res = experiments.run_method(obj, start, method, gradient_tolerance=1e-6, **kw)
        wall = float(res["trajectory"]["time_ms"][-1]) if len(res["trajectory"]["time_ms"]) else np.nan
        scale_rows.append(
            {"n": n, "метод": method, "итераций": res["iterations"],
             "время, мс": wall, "f*": res["value"], "сошёлся": res["converged"]}
        )
scale_table = pd.DataFrame(scale_rows)
scale_table.round(6)
"""
)
code(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
for idx, method in enumerate(scale_specs):
    sub = scale_table[scale_table["метод"] == method]
    axes[0].plot(sub["n"], sub["время, мс"], "o-", color=plotting.PALETTE[idx], label=method)
    axes[1].plot(sub["n"], sub["итераций"], "o-", color=plotting.PALETTE[idx], label=method)
axes[0].set_title("Время до сходимости vs размерность n")
axes[0].set_xlabel("n")
axes[0].set_ylabel("время, мс")
axes[0].set_yscale("log")
axes[0].legend()
axes[1].set_title("Число итераций vs размерность n")
axes[1].set_xlabel("n")
axes[1].set_ylabel("итераций")
axes[1].set_yscale("log")
axes[1].legend()
plt.show()
"""
)
md(
    r"""
**Вывод.** L-BFGS сохраняет качество BFGS при меньшей памяти и лучшем времени;
Adam требует на порядки больше итераций.
"""
)

# ---------------------------------------------------------------------------
# 6. Multistart robustness
# ---------------------------------------------------------------------------
md(
    r"""
## 6. Робастность на мультимодальной функции (multistart, Растригин)

12 случайных стартов на Rastrigin 2D. Метрика робастности — распределение
финального $f^*$ и число попаданий в область $f^*<1$.
"""
)
code(
    r"""
np.random.seed(2024)
starts = np.random.uniform(-5.12, 5.12, size=(12, 2))
multistart_methods = ["adam", "bfgs", "nelder_mead", "powell"]
ms_rows = experiments.multistart_compare(
    rastrigin, multistart_methods, starts=starts, max_iter=3000
)
ms_df = pd.DataFrame(ms_rows)

# Сводка по методам: лучший / медианный / худший f* и доля «успехов» (f* < 1).
summary = (
    ms_df.groupby("method")["value"]
    .agg(минимум="min", медиана="median", максимум="max")
    .reset_index()
)
summary["успехов (f*<1)"] = [
    int((ms_df[ms_df["method"] == m]["value"] < 1.0).sum())
    for m in summary["method"]
]
summary.columns = ["метод", "лучший f*", "медиана f*", "худший f*", "успехов из 12"]
summary.round(4)
"""
)
code(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

# (а) Boxplot распределения f* по методам.
data = [ms_df[ms_df["method"] == m]["value"].to_numpy() for m in multistart_methods]
bp = axes[0].boxplot(data, tick_labels=multistart_methods, patch_artist=True, showmeans=True)
for patch, color in zip(bp["boxes"], plotting.PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_title("Распределение финального f* (12 стартов)")
axes[0].set_ylabel("f*")
axes[0].tick_params(axis="x", rotation=15)

# (б) Доля успешных стартов.
success = [int((ms_df[ms_df["method"] == m]["value"] < 1.0).sum()) for m in multistart_methods]
plotting.bar_comparison(
    axes[1], multistart_methods, success,
    ylabel="успехов из 12", title="Попадания в глобальный минимум (f* < 1)",
)
axes[1].tick_params(axis="x", rotation=15)
plt.show()
"""
)
md(
    r"""
**Вывод.** На мультимодальной функции результат зависит от старта. Multistart
обязателен; Nelder-Mead здесь оказался самым робастным baseline.
"""
)

# ---------------------------------------------------------------------------
# 7. Sensitivity
# ---------------------------------------------------------------------------
md(
    r"""
## 7. Чувствительность к `learning_rate` и схеме градиента

Два чувствительных места first-order методов: `learning_rate` и источник
градиента. Проверяем GD по `lr` и Adam по `forward/central/five_point/autograd`.
"""
)
code(
    r"""
lr_rows = []
for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
    res = experiments.run_method(
        rosen, r_x0, "gradient_descent",
        learning_rate=lr, max_iter=5000, gradient_tolerance=1e-6,
        log_trajectory=False,
    )
    lr_rows.append({"lr": lr, "f*": res["value"], "iterations": res["iterations"],
                    "converged": res["converged"]})

scheme_rows = []
gradient_sources = {
    "forward": lambda x: optlib.RosenbrockNumericalGradient(x, scheme="forward"),
    "central": lambda x: optlib.RosenbrockNumericalGradient(x, scheme="central"),
    "five_point": lambda x: optlib.RosenbrockNumericalGradient(x, scheme="five_point"),
    "autograd": optlib.RosenbrockAutogradGradient,
}
for scheme, gradient in gradient_sources.items():
    res = optlib.Minimize(
        rosen.value, gradient, r_x0,
        method="adam", learning_rate=1e-2, max_iter=3000,
        gradient_tolerance=1e-6, step_tolerance=0.0,
        function_tolerance=0.0, log_trajectory=False,
    )
    scheme_rows.append({"gradient": scheme, "f*": res["value"], "iterations": res["iterations"],
                        "grad_norm": res["gradient_norm"], "converged": res["converged"]})

lr_table = pd.DataFrame(lr_rows)
scheme_table = pd.DataFrame(scheme_rows)
display(lr_table.round(8))
display(scheme_table.round(8))
"""
)
code(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].plot(lr_table["lr"], lr_table["f*"], "o-", color=plotting.PALETTE[0])
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("learning_rate")
axes[0].set_ylabel("f*")
axes[0].set_title("GD: чувствительность к learning_rate")

plotting.bar_comparison(
    axes[1], list(scheme_table["gradient"]), list(scheme_table["f*"] + 1e-16),
    ylabel="f*", title="Adam: источник градиента",
)
axes[1].set_yscale("log")
axes[1].tick_params(axis="x", rotation=15)
plt.show()
"""
)
md(
    r"""
**Вывод.** Малый `lr` слишком медленный, большой — нестабилен. `autograd` и
центральные схемы точнее forward; five-point дороже по вычислениям.
"""
)

# ---------------------------------------------------------------------------
# 8. Baseline vs scipy
# ---------------------------------------------------------------------------
md(
    r"""
## 8. Сравнение с эталоном `scipy.optimize`

Сравниваем наши реализации с `scipy.optimize` на Rosenbrock 2D.
"""
)
code(
    r"""
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scipy_rows = []
    for method in ("CG", "BFGS", "L-BFGS-B", "Newton-CG", "Nelder-Mead", "Powell"):
        row = experiments.scipy_minimize(rosen, r_x0, method=method, max_iter=20000)
        if row is not None:
            scipy_rows.append(row)

# Наши реализации тех же классов.
ours_methods = ["bfgs", "lbfgs", "newton", "nelder_mead", "powell"]
ours_rows = experiments.compare_methods(rosen, r_x0, ours_methods, max_iter=20000)

compare = pd.DataFrame(ours_rows + scipy_rows)[
    ["method", "value", "iterations", "function_evaluations",
     "gradient_evaluations", "converged"]
]
compare.columns = ["метод", "f*", "итераций", "выз. f", "выз. ∇f", "сошёлся"]
compare.round(10)
"""
)
md(
    r"""
**Вывод.** `optlib` воспроизводит качество `scipy` ($f^*\to0$) для second-order
и zero-order методов.
"""
)

# ---------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------
md(
    r"""
## Выводы

1. Second-order методы выигрывают на гладких задачах за счёт кривизны.
2. Zero-order методы нужны для black-box и разрывных функций.
3. Multistart обязателен на мультимодальных ландшафтах.
4. L-BFGS масштабируется лучше BFGS до $n=100$.
5. `optlib` совпадает с `scipy` по качеству на эталонных задачах.
"""
)

NB["cells"] = cells
NB["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = pathlib.Path(__file__).resolve().parent / "second_lab.ipynb"
nbf.write(NB, out)
print("wrote", out, "with", len(cells), "cells")
