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
