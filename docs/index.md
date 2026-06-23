# Документация optlib

`optlib` — учебная библиотека оптимизации на C++23 с Python-биндингами. Код
и API называются по-английски, а учебные материалы ведутся на русском языке.

## Разделы

- [Начало работы](getting-started.md) — установка, сборка и общий stage-gate.
- [Архитектура](architecture.md) — слои C++ ядра, биндингов и Python-пакета.
- [Линейная алгебра](linalg.md) — собственные контейнеры `Vector`/`Matrix` и
  горячие операции.
- [Дифференцирование](differentiation.md) — конечные разности и выбор шага.
- [Автодифференцирование](autograd.md) — multidual forward-mode AD.
- [Тестовые функции](functions.md) — Розенброк и аналитические производные.
- [Оптимизаторы](optimizers.md) — методы первого порядка и траектории.
- [Бенчмарки](benchmarks.md) — воспроизводимые измерения и внешние эталоны.
- [Нейронная сеть](neural-network.md) — MLP, BCE, backprop и flat parameters.
- [Датасеты](datasets.md) — d1/d2/d3, загрузка, preprocessing и F1.
- [Справочник API](api-reference.md) — публичные функции Python-пакета.

## Артефакты лабораторной 1

- `notebooks/first_lab.ipynb` строит траектории оптимизаторов на функции
  Розенброка, графики сходимости, сравнение схем дифференцирования и мини-бенч
  против `scipy.optimize`.
- `.agents/defense/lab1.md` содержит сценарий защиты и ссылки на эти разделы.
  Папка `.agents/` остается локальной и не коммитится по правилам проекта.

## Артефакты лабораторной 2

- `notebooks/second_lab.ipynb` сравнивает методы лабораторий 1-2 на Rosenbrock,
  Rastrigin, Himmelblau, Ackley и DesmosSurface.
- `docs/functions.md` описывает формулы, минимумы, размерности и гладкость
  benchmark-функций.
- `docs/benchmarks.md` фиксирует методологию multistart, метрики и SciPy
  baseline.

## Артефакты лабораторной 3

- `notebooks/third_lab.ipynb` обучает C++ MLP на d1/d2, сравнивает Adam и
  HeavyBall, строит loss-кривые, confusion matrix и границу решений d1.
- `docs/datasets.md` описывает загрузку d1/d2/d3, 80/20 split,
  стандартизацию и F1.
- `.agents/defense/lab3.md` содержит сценарий защиты и live-прогон d3.

## Артефакты лабораторной 4

- `notebooks/fourth_lab.ipynb` сравнивает оптимизаторы, schedules, stability по
  seed, L2/init ablation и sklearn/PyTorch baseline для MLP.
- `docs/neural-network.md` описывает MLP, backprop, weighted F1 и d3 fallback.
- `.agents/defense/lab4.md` содержит сценарий защиты и d3-команды.
