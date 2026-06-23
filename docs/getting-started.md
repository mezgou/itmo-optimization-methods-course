# Начало работы

`optlib` собирается как Python-пакет с C++23-расширением `_optlib`. На границе
Python/C++ используется pybind11, а сборкой управляют CMake и scikit-build-core.
Инструкции здесь не дублируются в `README.md`, потому что корневой файл оставлен
нетронутым по требованиям проекта.

## Окружение

Проект рассчитан на Python `>=3.12` и `uv`.

```powershell
uv sync --extra experiments --extra dev
uv pip install -e . --no-build-isolation
```

На Windows компилятор MSVC может быть доступен только после инициализации среды
Visual Studio. Если обычный PowerShell не видит `cl.exe`, запустите команды через
`VsDevCmd.bat` из установленной Visual Studio Build Tools или Community.

## Проверка этапа

Перед коммитом каждого этапа выполняется одинаковый gate:

```powershell
uv pip install -e . --no-build-isolation
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
cmake -B build -DOPTLIB_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Для запуска ноутбуков и сравнений со SciPy установите экспериментальные
зависимости:

```powershell
uv sync --extra experiments --extra dev
uv run jupyter notebook notebooks/first_lab.ipynb
```

## Текущая структура

- `src/optlib/` — C++23-ядро и pybind11-биндинги.
- `python/optlib/` — Python-пакет, который реэкспортирует `_optlib`.
- `tests/cpp/` — GoogleTest-тесты, запускаемые через CTest.
- `tests/python/` — pytest-тесты публичного Python API.
- `extends/pybind11` и `extends/googletest` — git submodule-зависимости.
- `notebooks/first_lab.ipynb` — воспроизводимый отчет первой лабораторной.
- `docs/` — русскоязычная документация по темам, а не по номерам лабораторных.

## Проверенные источники для сборки

Для первичной настройки использованы официальные материалы:

- [scikit-build-core](https://scikit-build-core.readthedocs.io/) — конфигурация
  задаётся в `pyproject.toml`, а backend вызывает CMake для сборки расширения.
- [pybind11 CMake helpers](https://pybind11.readthedocs.io/en/stable/cmake/) —
  `pybind11_add_module` создаёт Python extension module и подключает нужные
  Python-заголовки и флаги.
- [CMake GoogleTest module](https://cmake.org/cmake/help/latest/module/GoogleTest.html) —
  `gtest_discover_tests` регистрирует тесты из собранного исполняемого файла и
  удобен для CTest.
