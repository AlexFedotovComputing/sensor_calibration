import json
import re
from pathlib import Path

NB_PATH = Path('temperature_stability_modular_v2.ipynb')

def find_cell_idx(nb, startswith=None, contains=None):
    for i, c in enumerate(nb.get('cells', [])):
        src = ''.join(c.get('source', []))
        if startswith and src.lstrip().startswith(startswith):
            return i
        if contains and contains in src:
            return i
    return None

def ensure_all_figs_guard(code: str) -> str:
    guard = "if 'ALL_FIGS' not in globals():\n    ALL_FIGS = []\n"
    # insert guard after imports or at top
    lines = code.splitlines()
    inserted = False
    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        # place guard after a block of imports
        if not inserted and re.match(r"\s*import ", ln):
            # lookahead to next non-import
            if i+1 < len(lines) and not re.match(r"\s*import ", lines[i+1]):
                out.append(guard)
                inserted = True
    if not inserted:
        out.insert(0, guard)
    return '\n'.join(out)

def strip_all_figs_artifacts(code: str) -> str:
    # Remove any existing ALL_FIGS append lines and guards to keep idempotent
    lines = code.splitlines()
    out = []
    skip_next = False
    for ln in lines:
        if skip_next:
            skip_next = False
            continue
        if 'ALL_FIGS.append(' in ln:
            continue
        if "if 'ALL_FIGS' not in globals():" in ln:
            skip_next = True  # next line is usually initialization
            continue
        out.append(ln)
    code = '\n'.join(out)
    # Fix accidental double f-strings like ff'...'
    code = code.replace("ff'", "f'")
    return code

def add_fig_capture_after_show(code: str, title_expr: str, category: str) -> str:
    # Try to append capturing right before plt.show() or after fig creation
    lines = code.splitlines()
    out = []
    added_any = False
    for i, ln in enumerate(lines):
        out.append(ln)
        if re.search(r"plt\.show\(\)", ln):
            indent = re.match(r"^(\s*)", ln).group(1)
            out.append(f"{indent}ALL_FIGS.append(({category!r}, f{title_expr}, fig))")
            added_any = True
    if not added_any:
        # try add after first fig = plt.subplots(...)
        out2 = []
        inserted = False
        for ln in out:
            out2.append(ln)
            if not inserted and re.search(r"fig\s*,|fig\s*=", ln) and 'plt.subplots' in ln:
                indent = re.match(r"^(\s*)", ln).group(1)
                out2.append(f"{indent}ALL_FIGS.append(({category!r}, f{title_expr}, fig))")
                inserted = True
        out = out2
    return '\n'.join(out)

def inject_friendly_context(code: str) -> str:
    """Inject friendly sensor/ref names into plotting cells and use them in labels/titles."""
    lines = code.splitlines()
    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if re.search(r"for\s+sensor\s*,\s*g\s+in\s+calibration_points_by_sensor\.groupby\('\s*sensor\s*'\)\s*:", ln):
            out.append("    _nm = globals().get('SENSOR_NAMES', {})")
            out.append("    friendly = (_nm.get(sensor, sensor) if isinstance(_nm, dict) else sensor)")
            out.append("    _ref_disp = globals().get('REF_NAME', f'T{REF_IDX}')")
    code = '\n'.join(out)
    # Replace axis labels and titles to use friendly and ref display
    code = re.sub(r"set_xlabel\(f\'\{sensor\} \(X\)\'\)", "set_xlabel(f'{friendly} (X)')", code)
    code = re.sub(r"set_xlabel\(f\"\{sensor\} \(X\)\"\)", "set_xlabel(f'{friendly} (X)')", code)
    code = re.sub(r"set_ylabel\(f\'T\{REF_IDX\} \(Y\)\'\)", "set_ylabel(f'{_ref_disp} (Y)')", code)
    code = re.sub(r"set_ylabel\(f\"T\{REF_IDX\} \(Y\)\"\)", "set_ylabel(f'{_ref_disp} (Y)')", code)
    code = code.replace("f'Калибровочные точки: {sensor}'", "f'Калибровочные точки: {friendly}'")
    code = code.replace('f"Калибровка: {sensor}"', 'f"Калибровка: {friendly}"')
    code = code.replace("f'{sensor}: аппроксимация L2'", "f'{friendly}: аппроксимация L2'")
    code = code.replace("f'{sensor}: аппроксимация L_inf'", "f'{friendly}: аппроксимация L_inf'")
    return code

def restructure_report_cell(code: str) -> str:
    # Build a fresh template for clearer sections and collapsible procedure
    tpl = r'''# === 13) HTML-отчёт калибровки ===
import os, io, base64, datetime as _dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

assert 'calibration_points_by_sensor' in globals() and not calibration_points_by_sensor.empty, 'Нет калибровочной таблицы.'

REF_COL = f'T{REF_IDX}'
REF_DISPLAY = globals().get('REF_NAME', REF_COL)

# Имена датчиков: нормализация
_raw_names = globals().get('SENSOR_NAMES', {})
_name_map = {}
_ordered_cols = [f'T{REF_IDX+i}' for i in range(1, N_FOLLOW+1)]
if isinstance(_raw_names, (list, tuple)):
    for i, nm in enumerate(_raw_names, start=1):
        if i <= len(_ordered_cols):
            _name_map[_ordered_cols[i-1]] = str(nm)
elif isinstance(_raw_names, dict):
    for k, v in _raw_names.items():
        try:
            if isinstance(k, str) and k.startswith('T'):
                _name_map[k] = str(v)
            else:
                idx = int(k)
                if 1 <= idx <= len(_ordered_cols):
                    _name_map[_ordered_cols[idx-1]] = str(v)
        except Exception:
            pass
NAME_MAP = _name_map

def _fig_to_b64(fig):
    bio = io.BytesIO(); fig.savefig(bio, format='png', dpi=140, bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode('ascii')

def _html_escape(s):
    return (str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))

def _sensor_order(sensors):
    def key(c):
        try: return int(str(c).lstrip('T'))
        except: return 10**9
    return sorted(list(sensors), key=key)

expected = [f'T{REF_IDX+i}' for i in range(1, N_FOLLOW+1)]
_present_all = list(calibration_points_by_sensor['sensor'].unique())
present = [s for s in expected if s in _present_all] or _present_all
FOLLOW_ORDER = _sensor_order(present)

now = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
parts = []

# Заголовок
parts.append(f'<h1>Отчёт калибровки</h1>')
parts.append(f'<p>Время формирования: {now}</p>')
parts.append('<h2 id="toc">Содержание</h2>')
parts.append('<ol>')
parts.append('<li><a href="#sensors">Состав</a></li>')
parts.append('<li><a href="#explain">Пояснения по структуре</a></li>')
parts.append('<li><a href="#meas">Свойства калибровочного измерения</a></li>')
parts.append('<li><a href="#process">Процесс калибровки</a></li>')
parts.append('<li><a href="#models_all">Формулы и метрики (все степени)</a></li>')
parts.append('<li><a href="#approx">Детали аппроксимации</a></li>')
parts.append('<li><a href="#how">Как проводилась калибровка</a></li>')
parts.append('<li><a href="#physics">Физическая справка</a></li>')
parts.append('<li><a href="#appendix">Приложение: все графики</a></li>')
parts.append('</ol>')

# Состав сенсоров
parts.append('<h2 id="sensors">Состав</h2>')
parts.append('<ul>')
parts.append(f'<li>Эталон: {_html_escape(REF_DISPLAY)} ({_html_escape(REF_COL)})</li>')
for i, s in enumerate(FOLLOW_ORDER, start=1):
    friendly = NAME_MAP.get(s, s)
    parts.append(f"<li>Калибруемый {i}: {_html_escape(friendly)} ({_html_escape(s)})</li>")
parts.append('</ul>')

_mapping_items = []
for i, col in enumerate(FOLLOW_ORDER, start=1):
    _mapping_items.append(f"{i} → {_html_escape(NAME_MAP.get(col, col))} ({_html_escape(col)})")
parts.append('<p><i>Нумерация:</i> ' + '; '.join(_mapping_items) + '</p>')
parts.append('<h2 id="explain">Пояснения по структуре</h2>')
parts.append('<p>Отчёт состоит из: (1) свойств исходного измерения, (2) процесса калибровки (формирование X–Y и модели), (3) сводки формул и метрик для всех степеней, (4) деталей аппроксимации и (5) физической справки. В приложении — все ключевые графики.</p>')

# 1) Свойства калибровочного измерения
parts.append('<h2 id="meas">Свойства калибровочного измерения</h2>')

# Стабильные интервалы (если есть)
if 'STABLE_BY_SENSOR' in globals() and STABLE_BY_SENSOR is not None and not STABLE_BY_SENSOR.empty:
    parts.append('<h3>Стабильные интервалы (выбранные)</h3>')
    for s in FOLLOW_ORDER:
        disp = NAME_MAP.get(s, s)
        t = STABLE_BY_SENSOR[STABLE_BY_SENSOR['sensor']==s]
        if t.empty: continue
        td = t.copy(); td['start_date'] = pd.to_datetime(td['start_date']); td['end_date'] = pd.to_datetime(td['end_date'])
        tdd = td.reset_index(drop=True)
        import numpy as _np
        fig, ax = plt.subplots(figsize=(8, max(2, 0.3*len(tdd))))
        y = _np.arange(len(tdd))
        for i, row in tdd.iterrows():
            sdt = row['start_date']; edt = row['end_date']
            ax.hlines(i, sdt, edt, colors='C0', linewidth=6)
            ax.text(edt, i, f"  n={int(row['length'])}", va='center', fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels([f"seg {i+1}" for i in y])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.set_title(f'{_html_escape(disp)}: стабильные отрезки (выбранные)')
        ax.set_xlabel('Время'); ax.set_ylabel('Сегменты'); ax.grid(True, axis='x', alpha=0.3)
        fig.autofmt_xdate()
        parts.append(f'<p><img src="data:image/png;base64,{_fig_to_b64(fig)}" alt="{_html_escape(disp)} intervals"/></p>')

# 2) Процесс калибровки
parts.append('<h2 id="process">Процесс калибровки</h2>')

# Калибровочные точки X–Y
parts.append('<h3 id="xy">Калибровочные точки X–Y</h3>')
for s in FOLLOW_ORDER:
    g = calibration_points_by_sensor[calibration_points_by_sensor['sensor']==s].copy()
    disp = NAME_MAP.get(s, s)
    cols = ['bin_center','x_mean','y_mean','x_std','y_std','n_samples','start_date','end_date']
    for c in cols:
        if c not in g.columns: g[c] = np.nan
    parts.append(f'<h4>{_html_escape(disp)}</h4>')
    parts.append(g[cols].to_html(index=False, float_format=lambda v: f"{v:.6g}"))

# Формулы и метрики (все степени)
if 'calibration_models' in globals() and calibration_models is not None and not calibration_models.empty:
    parts.append('<h2 id="models_all">Формулы и метрики (все степени)</h2>')
    for s in FOLLOW_ORDER:
        disp = NAME_MAP.get(s, s)
        cm = calibration_models[calibration_models['sensor']==s]
        if cm.empty: continue
        cols = ['method','degree','formula','rmse','mae','maxerr']
        for c in cols:
            if c not in cm.columns: cm[c] = np.nan
        cm2 = cm[cols].sort_values(['method','degree'])
        parts.append(f'<h4>{_html_escape(disp)}</h4>')
        parts.append(cm2.to_html(index=False, escape=False, float_format=lambda v: f"{v:.6g}"))
else:
    parts.append('<p><i>Сначала выполните ячейку с моделями.</i></p>')

# Детали аппроксимации (методология)
parts.append('<h2 id="approx">Детали аппроксимации</h2>')
_use_scipy = False
try:
    import scipy  # noqa: F401
    _use_scipy = True
except Exception:
    _use_scipy = False
parts.append('<ul>')
parts.append('<li>L2 — взвешенная МНК (веса 1/max(y_std, ε) при наличии std эталона).</li>')
parts.append('<li>' + ('L_inf — минимакс через линейное программирование (SciPy HiGHS).' if _use_scipy else 'L_inf — минимакс, реализованный устойчивым IRLS‑методом.') + '</li>')
parts.append('<li>Степени перебираются от 1 до N−2 (где N — число калибровочных точек).</li>')
parts.append('<li>Калибровочные точки формируются из самых длинных стабильных интервалов в корзинах по DEG_TOL.</li>')
parts.append('</ul>')

# Справка (развернутая)
deg_tol = globals().get('DEG_TOL', None)
window_n = globals().get('WINDOW_N', None)
std_thr = globals().get('STD_THR', None)
diff_thr = globals().get('DIFF_THR', None)
min_len = globals().get('MIN_LEN', None)
max_ref_drift = globals().get('MAX_REF_DRIFT', None)
parts.append('<h2 id="how">Как проводилась калибровка</h2>')
parts.append('<ol>')
parts.append(f'<li>Формируются стабильные интервалы по эталону и датчикам: окно={window_n}, порог Std={std_thr}, порог средн. |Δ|={diff_thr}, мин. длина={min_len}.</li>')
parts.append(f'<li>Интервалы группируются по уровню эталона с шагом {deg_tol}°C, отбирается самый длинный в каждой корзине, контроль дрейфа эталона ≤ {max_ref_drift}°C.</li>')
parts.append('<li>По выбранным площадкам строятся точки X–Y (средние и std), где X — датчик, Y — эталон.</li>')
parts.append('<li>Подбираются полиномиальные модели L2 и L_inf для степеней 1..N−2; отчёт включает формулы и метрики для всех степеней.</li>')
parts.append('</ol>')

# Физическая справка
parts.append('<h2 id="physics">Физическая справка</h2>')
parts.append('<h3>Исходные данные</h3>')
parts.append('<p>Вход — журналы температур: столбец времени (date) и температурные каналы T0..T15 (в °C). Эталонный канал — ' + _html_escape(REF_COL) + ' (' + _html_escape(REF_DISPLAY) + '). Калибруемые каналы — ' + ', '.join(_html_escape(NAME_MAP.get(s, s)) for s in FOLLOW_ORDER) + '.</p>')
parts.append('<h3>Стабильные отрезки и калибровочные точки</h3>')
parts.append('<p>Стабильность определяется скользящими метриками по окну WINDOW_N: σ(T) ≤ STD_THR и средняя |ΔT| ≤ DIFF_THR. Эталон группируется по уровням с шагом DEG_TOL (корзины полочек), в каждой корзине берётся самый длинный стабильный интервал при контроле дрейфа эталона ≤ MAX_REF_DRIFT. Для этих интервалов рассчитываются средние и std: X = ⟨датчик⟩, Y = ⟨эталон⟩, а также x_std, y_std.</p>')
parts.append('<h3>Метрики качества</h3>')
parts.append('<ul>')
parts.append('<li><b>RMSE</b>: sqrt((1/N) ∑ (Y − Ŷ)^2)</li>')
parts.append('<li><b>MAE</b>: (1/N) ∑ |Y − Ŷ|</li>')
parts.append('<li><b>Max|err|</b>: max_i |Y_i − Ŷ_i|</li>')
parts.append('<li><b>Взвешивание</b>: при наличии y_std веса w_i = 1/max(y_std_i, ε)</li>')
parts.append('</ul>')
parts.append('<h3>Методы аппроксимации</h3>')
parts.append('<ul>')
parts.append('<li><b>L2</b>: взвешенная МНК (numpy.polyfit) — минимизирует ∑ w_i (Y_i − P(X_i))^2</li>')
parts.append('<li><b>L_inf</b>: минимакс — минимизация max_i w_i |Y_i − P(X_i)| через LP (SciPy HiGHS) либо устойчивый IRLS‑подход</li>')
parts.append('<li><b>Степень полинома</b>: перебор 1..N−2 (N — число калибровочных точек)</li>')
parts.append('</ul>')
parts.append('<h3>Интерпретация</h3>')
parts.append('<p>Практика: контролируйте Max|err| на краях диапазона, сравнивайте RMSE/MAE между L2 и L_inf, отслеживайте структуру остатков (смещения/немонотонности), избегайте переобучения при высоких степенях и слабом покрытии полочек.</p>')

# Приложение: все графики из ноутбука (если есть)
parts.append('<h2 id="appendix">Приложение: все графики из ноутбука</h2>')
if 'ALL_FIGS' in globals() and isinstance(ALL_FIGS, list) and ALL_FIGS:
    # group by category
    groups = {'measurement': [], 'process': [], 'other': []}
    for cat, title, fig in ALL_FIGS:
        key = cat if cat in groups else 'other'
        groups[key].append((title, fig))
    for cat_lbl, items in groups.items():
        if not items:
            continue
        title_map = {'measurement': 'Измерительные свойства', 'process': 'Процесс калибровки', 'other': 'Прочее'}
        parts.append(f"<h3>{title_map.get(cat_lbl, cat_lbl.title())}</h3>")
        for title, fig in items:
            parts.append(f"<p><b>{_html_escape(title)}</b></p>")
            parts.append(f"<p><img src='data:image/png;base64,{_fig_to_b64(fig)}' alt='{_html_escape(title)}'></p>")
else:
    parts.append('<p><i>Глобальный список ALL_FIGS пуст. Выполните ячейки с построением графиков.</i></p>')

html = (
    '<html><head><meta charset="utf-8"><title>Отчёт калибровки</title>'
    '<style>'
    'body{font-family:Segoe UI,Arial,sans-serif;line-height:1.35;color:#111} '
    'table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:4px 6px} '
    'h1{margin-top:0.2em} h2{margin-top:1.2em} h3{margin-top:0.9em} '
    'a{color:#0645ad;text-decoration:none} a:hover{text-decoration:underline} '
    '.muted{color:#666;font-size:90%} '
    '@media print{@page{size:A4;margin:15mm} h2{break-before:page} h3{break-inside:avoid} table{break-inside:avoid} img{max-width:100%;break-inside:avoid}} '
    '</style>'
    '</head><body>' + '\n'.join(parts) + '</body></html>'
)

out_path = os.path.join(os.getcwd(), 'calibration_report.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('HTML-отчёт сохранён:', out_path)
'''
    return tpl

def make_load_cell_self_contained(code_read_funcs: str, code_load: str) -> str:
    # Ensure pandas import present, and fallback read_one_table if missing
    pre = [
        'import pandas as pd',
        'import csv',
        'from typing import Optional',
        'from IPython.display import display',
        "import os, io",  # may already be present in code_load
        "# Fallback: define read functions if missing",
        "if 'read_one_table' not in globals():",
        *('    '+ln for ln in code_read_funcs.splitlines())
    ]
    # Merge, but avoid duplicate 'import os, io'
    merged = '\n'.join(pre) + '\n' + code_load
    # Make the load independent from REF_IDX/N_FOLLOW by keeping all T*-columns
    import re as _re
    merged = _re.sub(r"^\s*follow_idxs\s*=.*\n", "", merged, flags=_re.M)
    merged = _re.sub(r"^\s*cols_keep\s*=.*\n", "", merged, flags=_re.M)
    merged = _re.sub(r"^\s*data\s*=\s*data\[\[c for c in cols_keep.*\n", "", merged, flags=_re.M)
    merged = _re.sub(
        r"(\n\s*)DATA\s*=\s*data(?!\.)",
        "\n    t_cols = [c for c in data.columns if isinstance(c, str) and c.startswith('T')]\n    cols_keep = ['date'] + t_cols + (['source_file'] if 'source_file' in data.columns else [])\n    data = data[[c for c in cols_keep if c in data.columns]]\n\\1DATA = data",
        merged,
        count=0,
        flags=_re.M,
    )
    # Ensure combined_csv branch also filters columns
    merged = _re.sub(
        r"(t_cols\s*=\s*\[.*?\]\n)(\s*)DATA\s*=\s*data\.copy\(\)",
        r"\1\2cols_keep = ['date'] + t_cols + (['source_file'] if 'source_file' in data.columns else [])\n\2data = data[[c for c in cols_keep if c in data.columns]]\n\2DATA = data.copy()",
        merged,
        count=1,
        flags=_re.S,
    )
    # Avoid dependency on DATE_FORMAT when it's not yet defined
    merged = _re.sub(
        r"date_format=DATE_FORMAT\s*or\s*None",
        "date_format=(globals().get('DATE_FORMAT', '') or None)",
        merged,
        flags=_re.M,
    )
    # Deduplicate consecutive identical lines to prevent repeats across runs
    def _dedupe_consecutive(s: str) -> str:
        out=[]; prev=None
        for ln in s.splitlines():
            if ln==prev:
                continue
            out.append(ln); prev=ln
        return '\n'.join(out)
    merged = _dedupe_consecutive(merged)
    return merged

def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cells = nb['cells']

    # Locate relevant cells
    idx_load = find_cell_idx(nb, startswith="# === 6) Загрузка данных ===")
    if idx_load is None:
        idx_load = find_cell_idx(nb, contains="# === 6) Загрузка данных ===")
    idx_read = find_cell_idx(nb, startswith="# === 2) Загрузка/парсинг ===")
    idx_plot_quick = find_cell_idx(nb, startswith="# === 10) Быстрый график калибровочных точек с ошибками ===")
    idx_plot_poly = find_cell_idx(nb, startswith="# === 11) Калибровочные кривые (полиномы) ===")
    idx_models = find_cell_idx(nb, startswith="# === 12) Калибровочные модели (L2 и L_inf) ===")
    idx_report = find_cell_idx(nb, startswith="# === 13) HTML-отчёт калибровки ===")

    # 1) Make plotting cells register figures to ALL_FIGS
    for idx, title_expr, cat in [
        (idx_plot_quick, "'{friendly}: точки и std'", 'measurement'),
        (idx_plot_poly, "'{friendly}: полиномы и остатки'", 'process'),
        (idx_models, "'{friendly}: L2/L_inf — кривые и остатки'", 'process'),
    ]:
        if idx is not None:
            src = ''.join(cells[idx].get('source', []))
            # strip previous artifacts to avoid duplicates across runs
            src = strip_all_figs_artifacts(src)
            # inject friendly names for labels/titles
            src = inject_friendly_context(src)
            src = ensure_all_figs_guard(src)
            src = add_fig_capture_after_show(src, title_expr, cat)
            cells[idx]['source'] = [l+('\n' if not l.endswith('\n') else '') for l in src.splitlines()]

    # 2) Restructure report cell
    if idx_report is not None:
        new_report = restructure_report_cell('')
        nb['cells'][idx_report]['source'] = [l+('\n' if not l.endswith('\n') else '') for l in new_report.splitlines()]

    # 3) Move load cell to be the first code cell (index 1 overall), and make it self-contained
    if idx_load is not None:
        load_src = ''.join(cells[idx_load].get('source', []))
        read_src = ''.join(cells[idx_read].get('source', [])) if idx_read is not None else ''
        # Extract only functions from read_src (simple cut: find 'def sniff_sep' and take until end)
        m = re.search(r"def\s+sniff_sep\(.*?\):[\s\S]*?def\s+read_one_table\(.*?\):[\s\S]*", read_src)
        read_funcs = m.group(0) if m else read_src
        new_load = make_load_cell_self_contained(read_funcs, load_src)
        cells[idx_load]['source'] = [l+('\n' if not l.endswith('\n') else '') for l in new_load.splitlines()]

        # Reorder: place load cell right after the first markdown (index 0)
        load_cell = cells.pop(idx_load)
        cells.insert(1, load_cell)

        # Update in nb
        nb['cells'] = cells

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Notebook updated.')

if __name__ == '__main__':
    main()
