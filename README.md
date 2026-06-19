# LJ-флюид и уравнение Ван-дер-Ваальса в Colab

Это самодостаточный вычислительный ноутбук для небольшой учебно-исследовательской
работы по молекулярной динамике. Проект не является библиотекой или
программным фреймворком: основной рабочий артефакт один — `lj_vdw_colab.ipynb`.

## Что делает ноутбук

1. Создаёт периодическую 3D-систему частиц Lennard-Jones в OpenMM.
2. Запускает NVT-моделирование с Langevin-термостатом.
3. Считает EOS-точки для сетки температур, плотностей и seed.
4. Сохраняет две таблицы:
   - `data/eos_points.csv`
   - `data/eos_final_profiles.csv`
5. Подбирает параметры `a,b` уравнения Ван-дер-Ваальса:

       P = rho*T/(1 - b*rho) - a*rho^2

6. Строит графики:
   - `figures/eos_isotherms.png`
   - `figures/vdw_fit.png`
   - `figures/vdw_residuals.png`
7. Архивирует `data/` и `figures/` в `lj_vdw_results.zip` для скачивания из Colab.

## Как открыть в Google Colab

Откройте ссылку:

<https://colab.research.google.com/github/Bettta11/VPV_LJ/blob/colab-single-notebook/lj_vdw_colab.ipynb>

Перед запуском включите GPU:

1. `Runtime -> Change runtime type`
2. `Hardware accelerator -> GPU`
3. `Save`

Затем выполняйте ячейки сверху вниз.

## Первая ячейка

Первая рабочая ячейка сама устанавливает зависимости:

```python
openmm
numpy
pandas
scipy
matplotlib
tqdm
```

После установки она выводит список доступных OpenMM-платформ и явно показывает,
видна ли `CUDA`. Если `CUDA` не видна, ноутбук не падает сразу, но предупреждает,
что расчёт пойдёт на CPU и может быть медленным.

## Где менять параметры

Все параметры расчёта находятся в одной ячейке `PARAMS`. Там задаются:

- `N`
- `sigma`, `epsilon`, `mass`
- `rcut`
- `dt`
- `equil_steps`, `prod_steps`, `sample_interval`
- `friction`
- `temperatures`
- `densities`
- `seeds`
- `profile_bins`
- настройки OpenMM platform

Чтобы изменить расчёт, просто отредактируйте эту ячейку и перезапустите ноутбук.

## Ограничения постановки

- Нет slab-геометрии.
- Нет гравитации и стенок.
- Основная EOS-часть не сохраняет траектории.
- Не создаётся отдельная папка на каждую EOS-точку.
- Не создаётся `eos_morphology.csv`.
- `eos_final_profiles.csv` хранит только counts по z-бинам с колонками:
  `run_id, bin, z_min, z_max, z_center, count`.

Такой формат сделан специально для понятной студенческой вычислительной работы:
LJ-модель -> EOS-таблицы -> fit Ван-дер-Ваальса -> графики области применимости.

## GPU-бенчмарки перед большим запуском

Отдельный ноутбук `lj_vdw_benchmarks.ipynb` нужен для подготовки будущего
production-run в Google Colab или Yandex DataSphere на GPU. Он проверяет, видит
ли OpenMM платформу `CUDA`, измеряет скорость для разных размеров системы,
проверяет стратегию multi-GPU как набор независимых worker-ов по разным EOS-точкам
и помогает выбрать разумные `N`, `equil_steps` и `prod_steps`.

Запускать его нужно сверху вниз в GPU-окружении Colab/DataSphere. Первая ячейка
сама устанавливает зависимости, выводит `nvidia-smi`, список OpenMM platforms и
явно предупреждает, если расчёт идёт на CPU. Все параметры находятся в одной
ячейке `BENCHMARK_PARAMS`; внешних конфигов нет.

Этот ноутбук не строит финальные EOS-графики, не фитит уравнение
Ван-дер-Ваальса, не сохраняет траектории, координаты, профили плотности,
картинки или видео. Его результат — только benchmark-таблицы в `benchmark_data/`:

- `speed_benchmark.csv`
- `multigpu_benchmark.csv`
- `equilibration_traces.csv`
- `equilibration_blocks.csv`
- `benchmark_summary.csv`
- `runtime_estimates.csv`

По `speed_benchmark.csv` выбирают практичный размер системы и оценивают время на
одной GPU. По `multigpu_benchmark.csv` проверяют, есть ли выигрыш от запуска
нескольких независимых EOS-точек на разных GPU. По `equilibration_blocks.csv` и
`benchmark_summary.csv` выбирают консервативные `equil_steps` и `prod_steps`.
`runtime_estimates.csv` даёт предварительную оценку времени для будущего
облачного acquisition notebook, который должен снимать только EOS-точки,
поддерживать resume, писать shard-файлы по worker/GPU и сохранять failures без
остановки всей сетки.
