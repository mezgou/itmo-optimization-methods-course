# Документация optlib

`optlib` - библиотека оптимизации на C++23 с Python-пакетом `optlib` и
расширением `_optlib` на pybind11. C++ ядро содержит собственные реализации
линейной алгебры, дифференцирования, оптимизаторов и небольшой MLP для бинарной
классификации. Python слой нужен для воспроизводимых экспериментов, графиков,
работы с CSV-датасетами и ноутбуками.

Документация описывает текущее устройство проекта, а не историю разработки.
Код, имена API и идентификаторы остаются на английском; пояснения написаны на
русском.

## Разделы

- [Начало работы](getting-started.md): установка, сборка, тесты, запуск ноутбуков.
- [Сборочная система](build-system.md): pyproject, scikit-build-core, CMake options.
- [Архитектура](architecture.md): слои проекта, ownership данных, GIL, структура каталогов.
- [Справочник API](api-reference.md): публичные функции `optlib` и возвращаемые структуры.
- [Линейная алгебра](linalg.md): `Vector`, `Matrix`, BLAS-подобные операции ядра.
- [Численное дифференцирование](differentiation.md): конечные разности и выбор шага.
- [Автоматическое дифференцирование](autograd.md): forward-mode dual numbers.
- [Тестовые функции](functions.md): Rosenbrock, Rastrigin, Himmelblau, Ackley, DesmosSurface и другие.
- [Оптимизаторы](optimizers.md): методы первого, второго и нулевого порядка.
- [Нейронная сеть](neural-network.md): бинарная MLP, BCE, backprop, F1.
- [Датасеты](datasets.md): CSV-формат, d1/d2, закрытый d3, стандартизация и оценка.
- [Бенчмарки и отчеты](benchmarks.md): методика сравнений, notebooks, внешние baseline.

## Основные артефакты

- `src/optlib/core/` - C++23 ядро без Eigen, BLAS, Boost и CUDA.
- `src/optlib/bindings/Module.cpp` - pybind11-граница и преобразование NumPy массивов.
- `python/optlib/` - Python wrappers, dataset pipeline, plotting и study helpers.
- `tests/cpp/` - GoogleTest проверки ядра.
- `tests/python/` - pytest проверки Python API и воспроизводимых сценариев.
- `notebooks/first_lab.ipynb`, `second_lab.ipynb`, `third_lab.ipynb`, `fourth_lab.ipynb` -
  готовые отчеты с графиками и таблицами.
- `data/first_dataset.csv`, `data/second_dataset.csv` - открытые CSV-датасеты.
- `scripts/download_dataset.py` - CLI для загрузки CSV по Google Drive id.

## Качество и воспроизводимость

Перед публикацией изменений проверяются сборка расширения, Python-тесты, ruff и
C++ тесты:

```powershell
uv pip install -e . --no-build-isolation
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
ctest --test-dir build
```

Ноутбуки строятся из генераторов `notebooks/_build_*_lab.py`. Это удерживает
единый стиль, фиксированные seed и повторяемую структуру отчетов.
