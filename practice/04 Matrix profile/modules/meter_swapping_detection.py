import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions: dict, cutoff, house_idx: list) -> tuple:
    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads: dict, tails: dict, house_idx: list, m: int) -> dict:
    eps = 1e-6
    
    min_global_score = np.inf
    best_match = {'score': np.inf, 'i': None, 'j': None, 'mp_j': None}

    # Функция-помощник для безопасного извлечения минимального расстояния
    def get_min_finite_distance(ts1, ts2, m):
        if len(ts1) < m or len(ts2) < m:
            return np.inf
        
        # Убедимся, что данные чистые (без NaN) перед передачей в stumpy
        ts1 = np.nan_to_num(ts1, nan=np.inf)
        ts2 = np.nan_to_num(ts2, nan=np.inf)

        # Вычисляем профиль
        mp_dict = compute_mp(ts1=ts1, m=m, ts2=ts2)
        mp_values = mp_dict['mp'].astype(np.float64)
        
        # Находим минимальное значение среди конечных чисел (игнорируем inf и nan)
        finite_vals = mp_values[np.isfinite(mp_values)]
        
        if len(finite_vals) > 0:
            return np.min(finite_vals), mp_dict
        else:
            return np.inf, mp_dict

    # 1. Рассчитываем "базовые" дистанции
    baseline_distances = {}
    for i in house_idx:
        head_i = heads[f'H_{i}'].values.flatten()
        tail_i = tails[f'T_{i}'].values.flatten()
        baseline_distances[i], _ = get_min_finite_distance(head_i, tail_i, m)

    # 2. Перебираем все пары и считаем score
    for i in house_idx:
        for j in house_idx:
            head_i = heads[f'H_{i}'].values.flatten()
            tail_j = tails[f'T_{j}'].values.flatten()
            
            min_dist_ij, mp_ij_dict = get_min_finite_distance(head_i, tail_j, m)
            
            baseline_dist = baseline_distances.get(i, np.inf)
            
            # Пропускаем, если одно из расстояний бесконечно
            if np.isinf(min_dist_ij) or np.isinf(baseline_dist):
                continue

            score = min_dist_ij / (baseline_dist + eps)

            if score < min_global_score:
                min_global_score = score
                best_match = {
                    'score': min_global_score, 
                    'i': i, 
                    'j': j, 
                    'mp_j': mp_ij_dict
                }
    return best_match


def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")
