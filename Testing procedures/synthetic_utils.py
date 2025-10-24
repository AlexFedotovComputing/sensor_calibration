import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

'''
В реальности при выставлении температуры на калибраторе вместо постоянного T
наблюдаем "переходное", изменяющееся со временем по закону вида T + T_var(t), где
T_var(t) = A * exp(-alpha * t) * sin(2 * pi * (t / N))
Сигнал, снимаемый датчиками, может искажаться за счет:  
noise - высокочастотного шума 
bias - постоянного сдвига среднего относительно Т, влияет на mean
drift - динамического сдвига во времени отосительно Т 
gain - мультипликативного коэффициента, изменяющего амплитуду сигнала, и, как следствие, std
Таким образом, датчики фиксируют T + drift + bias + gain * T_var(t) + noise
Upd: добавлен ramp - линейный переход от одного Т к следующему
'''

def make_synthetic_df(bigsteps=[19,23,27,31,35], # значения температуры, выставляемые на калибраторе (большие ступеньки)
                                  bigstep_duration=[1800,1800,1800,1800,1800], # длительности больших ступенек
                                  sampling_interval=1.0, # частота логирования (в секундах)
                                  decay_rate=0.05, # alpha в показателе экспоненты T_var(t)
                                  n_sin_periods=60.0, #  N число периодов синуса на временном отрезке t в T_var(t)
                                  amplitude=0.5, # амплитуда A в T_var(t)
                                  sensor_sigmas=None, # sigma шума на каждом датчике, dict {'T8':0.01,...} или scalar
                                  sensor_biases=None, # bias на каждом датчике, dict {'T8':0.01,...} или scalar
                                  sensor_gains=None, # gain на каждом датчике, dict {'T8':0.01,...} или scalar
                                  sensor_drifts=None, # drift на каждом датчике, dict {'T8':0.01,...} или scalar
                                  ramp_time=200.0, # время линенйого перехода от одного Т к следующему
                                  nan_prob=0.0, seed=None, source_name='SYNTH'):
    '''
    Функция генерирует синтетический датасет-лог для тестов калибровки.
    Возвращает DataFrame со столбцами date, T0..T15, source_file.
    По умолчанию (для экономии памяти) заполняются только столбцы T8..T11.
    Т8 - столбец эталон.
    '''
    rng = np.random.default_rng(seed)
    sensor_sigma_default = 0.001
    sensor_bias_default = 0.0
    sensor_gain_default = 1.0
    sensor_drift_default = 0.0

    # собираем параметры сигналов для датчиков Т8..Т11
    sensors = [8,9,10,11]
    sensor_keys = [f'T{i}' for i in sensors]

    def normalize_param(param, default):
        if param is None or isinstance(param, (int, float)):
            val = default if param is None else param
            return {k: val for k in sensor_keys}
        if isinstance(param, dict):
            return {k: param.get(k, default) for k in sensor_keys}
        raise TypeError("Параметр должен быть None, числом или dict")

    sensor_sigmas = normalize_param(sensor_sigmas, sensor_sigma_default)
    sensor_biases = normalize_param(sensor_biases, sensor_bias_default)
    sensor_gains = normalize_param(sensor_gains, sensor_gain_default)
    sensor_drifts = normalize_param(sensor_drifts, sensor_drift_default)
    
    # находим число сэмплов
    samples_per_step = [int(max(1, round(s/sampling_interval))) for s in bigstep_duration]
    total = sum(samples_per_step)
    # строим колонку с датами
    start = datetime(2025,1,1,0,0,0)
    times = [start + timedelta(seconds=i*sampling_interval) for i in range(total)]
    data = { 'date': times }
    # заполняем столбцы T0..T15 NaN'ами
    for i in range(16):
        data[f'T{i}'] = np.full(total, np.nan, dtype=float)

    # генерируем сигналы
    pos = 0 # глобальный индекс шага по времени
    prev_bigstep = None
    ramp_samples = int(ramp_time / sampling_interval) if ramp_time and ramp_time > 0 else 0
    for step_idx, (bigstep, n_samples) in enumerate(zip(bigsteps, samples_per_step)):

                
        pos_local = np.arange(n_samples) # локальный индекс шага по времени на текущей bigstep
        ramp_len = min(ramp_samples, n_samples) if ramp_samples > 0 else 0

        # вычисляем base_temp с учетом ramp
        if prev_bigstep is None:
            base_temps = np.full(n_samples, bigstep, dtype=float)
        else:
            base_temps = np.full(n_samples, bigstep, dtype=float)
            if ramp_len > 0:
                ramp_indices = pos_local[:ramp_len]
                base_temps[:ramp_len] = prev_bigstep + (ramp_indices + 1) * (bigstep - prev_bigstep) / ramp_samples

        # вычисляем "переходную" компоненту T_var(t)
        t_rel = pos_local - ramp_len
        # transient = amplitude * np.exp(-decay_rate * pos_local) * np.sin(2 * np.pi * (pos_local / n_sin_periods))
        transient = np.zeros(n_samples, dtype=float)
        mask_after_ramp = t_rel >= 0
        if mask_after_ramp.any():
            t_rel_positive = t_rel[mask_after_ramp]
            transient[mask_after_ramp] = (
                amplitude * np.exp(-decay_rate * t_rel_positive) * np.sin(2 * np.pi * (t_rel_positive / n_sin_periods)))

        # глобальные индексы (нужны для дрейфа)
        global_indices = np.arange(pos, pos + n_samples)
        for s in sensors:
            key = f'T{s}'
            sigma = sensor_sigmas[key]
            bias = sensor_biases[key]
            gain = sensor_gains[key]
            drift_per_sample = sensor_drifts[key]
            # при переходе к новой ступеньке дрейф не сбрасывается, а продолжает накапливаться,
            # поэтому умножаем на глобальный индекс
            drift = drift_per_sample * global_indices
            noise = rng.normal(0, sigma, n_samples)
            val = base_temps + drift + bias + gain * transient + noise
            data[key][pos : pos + n_samples] = val
        prev_bigstep = bigstep
        pos += n_samples
        
    # optionally introduce NaNs randomly
    if nan_prob and nan_prob>0.0:
        total_points = total * len(sensors)
        # for reproducibility pick positions
        for key in [f'T{i}' for i in sensors]:
            mask = rng.random(total) < nan_prob
            arr = np.array(data[key], dtype=float)
            arr[mask] = np.nan
            data[key] = arr

    df = pd.DataFrame(data)
    df['source_file'] = source_name
    return df

def plot_sensor_readings(df, sensors=None, time_col='date', title='Sensor readings'):
    '''
    Строит показания датчиков из DataFrame df.
    '''
    if sensors is None:
        sensors = [c for c in df.columns if c.startswith('T') and c[1:].isdigit()]
        sensors = sorted(sensors, key=lambda x: int(x[1:]))

    fig, ax = plt.subplots(figsize=(12, 6))
    for key in sensors:
        ax.plot(df[time_col], df[key], label=key)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S\n%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def save_synthetic_df(df, base_filename='synthetic_data'):
    '''
    Для сохранения датасета в формате csv.
    '''
    csv_filename = f"{base_filename}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"DataFrame сохранен в: {csv_filename}")
    