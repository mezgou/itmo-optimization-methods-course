# Сборочная система

Проект собирается как Python-пакет с C++23 расширением. Главная точка входа для
Python packaging - `pyproject.toml`, для C++ - `CMakeLists.txt`.

## pyproject.toml

Build backend:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.13"]
build-backend = "scikit_build_core.build"
```

Пакет:

- имя: `optlib`;
- версия: `1.0.0`;
- Python: `>=3.12`;
- runtime dependency: `numpy>=2.0`.

Extra `dev` добавляет pytest, ruff, scikit-build-core и pybind11. Extra
`experiments` добавляет SciPy, matplotlib, pandas, scikit-learn, torch, Jupyter,
ipykernel и gdown.

## scikit-build-core

Editable install:

```powershell
uv pip install -e . --no-build-isolation
```

scikit-build-core вызывает CMake и кладет Python wheel package из
`python/optlib`. Расширение `_optlib` устанавливается в пакет `optlib`.

## CMake options

Основные опции:

- `OPTLIB_BUILD_TESTS=ON` - собрать GoogleTest executable.
- `OPTLIB_BUILD_BENCHMARKS=ON` - собрать C++ microbenchmark.
- `OPTLIB_OPENMP=OFF` - зарезервированная опция; текущий код не требует OpenMP.

Сборка:

```powershell
cmake -B build -DOPTLIB_BUILD_TESTS=ON -DOPTLIB_BUILD_BENCHMARKS=ON
cmake --build build --config Release
ctest --test-dir build
```

## Vendored dependencies

`extends/pybind11` подключается через `add_subdirectory`. GoogleTest подключается
только при `OPTLIB_BUILD_TESTS=ON`.

C++ ядро не использует внешние математические библиотеки. Это делает сборку
предсказуемой и упрощает проверку алгоритмов.

## Оптимизации

Для MSVC используются `/O2` и `/permissive-`. Для non-MSVC Release-сборки
используется `-O3`, а `-march=native` добавляется только если компилятор
поддерживает этот флаг.

## Проверочный минимум

```powershell
uv pip install -e . --no-build-isolation
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
ctest --test-dir build
```
