# Начало работы

## Требования

Проект рассчитан на Windows и установленный C++ toolchain с поддержкой C++23.
На Windows достаточно MSVC из Visual Studio Build Tools. Дополнительный clang не
нужен.

Нужны:

- `uv` для Python-окружения;
- CMake 3.27 или новее;
- компилятор C++23;
- vendored `pybind11` и GoogleTest из `extends/`.

C++ зависимости ограничены pybind11 и GoogleTest. Линейная алгебра,
дифференцирование, оптимизаторы и MLP реализованы внутри `optlib`.

## Установка окружения

Базовая разработческая установка:

```powershell
uv sync --extra dev
uv pip install -e . --no-build-isolation
```

Для ноутбуков и внешних baseline:

```powershell
uv sync --extra dev --extra experiments
uv pip install -e . --no-build-isolation
```

Проверка импорта:

```powershell
uv run python -c "import optlib; print(optlib.__version__)"
```

## Сборка C++

Обычная Release-сборка:

```powershell
cmake -B build -DOPTLIB_BUILD_TESTS=ON -DOPTLIB_BUILD_BENCHMARKS=ON
cmake --build build --config Release
ctest --test-dir build
```

scikit-build-core использует CMake при сборке editable пакета, поэтому команда
`uv pip install -e . --no-build-isolation` также пересобирает `_optlib`.

## Проверка качества

Минимальный набор проверок:

```powershell
uv pip install -e . --no-build-isolation
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
ctest --test-dir build
```

Если `build/` еще не создан, перед `ctest` выполните CMake-команды из раздела
выше.

## Запуск ноутбуков

Готовые отчеты лежат в `notebooks/`. Для интерактивного запуска:

```powershell
uv run jupyter notebook notebooks/first_lab.ipynb
uv run jupyter notebook notebooks/second_lab.ipynb
uv run jupyter notebook notebooks/third_lab.ipynb
uv run jupyter notebook notebooks/fourth_lab.ipynb
```

Пересборка ноутбуков из генераторов:

```powershell
uv run python notebooks/_build_first_lab.py
uv run python notebooks/_build_second_lab.py
uv run python notebooks/_build_third_lab.py
uv run python notebooks/_build_fourth_lab.py
```

## Данные

Открытые CSV лежат в `data/`. Закрытый CSV можно скачать по Google Drive id:

```powershell
uv run python scripts/download_dataset.py <file_id> data/third_dataset.csv
```

То же из Python:

```python
import optlib

optlib.download("<file_id>", "data/third_dataset.csv")
```
