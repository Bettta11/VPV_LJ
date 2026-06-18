# VPV_LJ

Компактный OpenMM-проект для моделирования Lennard-Jones-флюида и проверки
применимости уравнения Ван-дер-Ваальса по данным молекулярной динамики.

## Статус

- Этап 1: каркас проекта, общий OpenMM-core и короткий debug-запуск готовы.
- Этап 2: EOS sweep, eos_points.csv, eos_final_profiles.csv и первичные графики готовы.
- Этап 3: fit уравнения Ван-дер-Ваальса и графики сравнения готовы.
- Этап 4: visual workflow и полный управляющий ноутбук пока не реализованы.

## Структура

    cloud_runner.ipynb
    core.py
    debug.py
    eos.py
    visual.py
    configs/
    data/
    report_assets/

## Debug-проверка

    .venv/bin/python debug.py configs/debug.yaml

Ожидаемые файлы:

- data/debug/debug_001/config.yaml
- data/debug/debug_001/state_trace.csv
- data/debug/debug_001/trajectory.dcd
- data/debug/debug_001/topology.pdb

## EOS-запуск

    .venv/bin/python eos.py configs/eos.yaml

EOS-режим складывает все точки одной серии в одну директорию, например
data/eos/eos_001/.

Создаваемые файлы:

- config.yaml
- log.txt
- eos_points.csv
- eos_final_profiles.csv
- figures/eos_isotherms.png
- figures/eos_energy.png
- figures/profile_overview.png

## Анализ Ван-дер-Ваальса

После EOS-запуска можно подобрать параметры `a,b` по уже существующей таблице
`eos_points.csv`:

    .venv/bin/python eos.py --fit-vdw data/eos/eos_001

Fit использует модель:

    P = rho*T/(1 - b*rho) - a*rho^2

По умолчанию fit-region явно задан как все конечные строки `eos_points.csv` со
`status=ok`; использованные строки и описание области сохраняются в
`vdw_fit.json`.

Создаваемые файлы:

- vdw_fit.json
- figures/vdw_fit.png
- figures/vdw_residuals.png
- figures/vdw_temperature_series.png

Ограничения EOS-режима:

- не используется гравитация: external_field.type: none, g: 0.0;
- не сохраняются траектории;
- не создаются отдельные папки для отдельных точек сетки;
- не создаётся eos_morphology.csv;
- eos_final_profiles.csv хранит только run_id, bin, z_min, z_max, z_center, count.

Сгенерированные данные, траектории, виртуальное окружение и кэши игнорируются Git.
