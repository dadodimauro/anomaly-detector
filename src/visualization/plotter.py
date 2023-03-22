import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def start_renderer(rederer='sphinx_gallery'):
    pio.renderers.default = rederer # 'jupyterlab' or 'notebook' or 'colab' or 'iframe' or 'iframe_connected' or 'sphinx_gallery'
    
    
def plot_history(history, save_static=False, save_html=False, path='./reports/figures/', filename='history'):
    start_renderer()
    
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, len(losses1)-1, len(losses1)), y=losses1,
                        mode='lines',
                        name='loss1'))
    fig.add_trace(go.Scatter(x=np.linspace(0, len(losses2)-1, len(losses2)), y=losses2,
                        mode='lines',
                        name='loss2'))
    
    fig.update_yaxes(title_text="loss")
    fig.update_yaxes(title_text="epoch")

    fig.update_layout(
        title_text='Losses vs. No. of epochs'
    )
    
    if save_static is True:
        if not os.path.exists(path + 'history'):
            os.makedirs(path + 'history')
        if not os.path.exists(path + 'history/static'):
            os.makedirs(path + 'history/static')
  
        full_path = path + "history/static/" + filename 
        fig.write_image(full_path + '.svg')
        
    if save_html is True:
        if not os.path.exists(path + 'history'):
            os.makedirs(path + 'history')
        if not os.path.exists(path + 'history/dynamic'):
            os.makedirs(path + 'history/dynamic')
  
        full_path = path + "history/dynamic/" + filename 
        fig.write_html(full_path + '.html')
    
    fig.show()


def plot_res_db_time(res, db_time, timestamps=None, th=None, title='', 
                                                             save_static=False, save_html=False, 
                                                             path='./reports/figures/', filename='db_time'):
    
    offset = len(db_time) - len(res)  # the offset is the len of the window
    x_len = len(res)
    if timestamps is None:
        x = np.linspace(0, x_len-1, x_len)
    else:
        x = timestamps[offset:]
    
    start_renderer()
    
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing = 0.05
    )

    fig.add_trace(
        go.Scatter(x=x, y=res, name='res'),
        row=1, col=1
    )
    
    if th is not None:
        fig.add_hline(y=th, line_dash="dot", row=1, col=1, line_width=2, name='threshold')

    fig.add_trace(
        go.Scatter(x=x, y=db_time[offset:], name='db time'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400, width=1200,
        title_text=title
    )
    
    if save_static is True:
        if not os.path.exists(path + 'db_time'):
            os.makedirs(path + 'db_time')
        if not os.path.exists(path + 'db_time/static'):
            os.makedirs(path + 'db_time/static')
  
        full_path = path + "db_time/static/" + filename 
        fig.write_image(full_path + '.svg')
        
    if save_html is True:
        if not os.path.exists(path + 'db_time'):
            os.makedirs(path + 'db_time')
        if not os.path.exists(path + 'db_time/dynamic'):
            os.makedirs(path + 'db_time/dynamic')
  
        full_path = path + "db_time/dynamic/" + filename 
        fig.write_html(full_path + '.html')
    
    fig.show()
    
    
# USING PYPLOT
# def plot_res_db_time(res, db_time, th=None, title=''):
#     # create two subplots with the shared x and y axes
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10,3))
#     ax1.plot(res, label='res')
#     if th is not None:
#         ax1.axhline(y=th, color='r')
#     ax1.legend()
#     ax1.grid()
#     ax2.plot(db_time, label='db_time')
#     ax2.legend()
#     ax2.grid()
#     plt.suptitle(title)
#     plt.show()

def plot_labels(y_pred, labels, db_time=None, timestamps=None, th=None, title='',
                                                                        save_static=False, save_html=False,
                                                                        path='./reports/figures/', filename='labels'):
    
    x_len = len(y_pred)
    if timestamps is None:
        x = np.linspace(0, x_len-1, x_len)
    else:
        offset = len(timestamps) - len(y_pred)  # the offset is the len of the window
        x = timestamps[offset:]
    
    start_renderer()
    
    if db_time is None:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=x, y=y_pred, name='y_pred'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=x, y=labels, name='labels'),
            secondary_y=True
        )
        # Set y-axes titles
        fig.update_yaxes(title_text="prediction score", secondary_y=False)
        
        
    else:
        fig = make_subplots(
            rows=1, cols=2,
            horizontal_spacing = 0.05
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_pred, name='y_pred'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=labels*y_pred, name='labels'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=db_time[offset:], name='db time'),
            row=1, col=2
        )
        # Set y-axes titles
        fig.update_yaxes(title_text="prediction score", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="db time", row=1, col=2)
        
    fig.update_layout(
        height=400, width=1200,
        title_text=title,
    )
    
    if save_static is True:
        if not os.path.exists(path + 'labels'):
            os.makedirs(path + 'labels')
        if not os.path.exists(path + 'labels/static'):
            os.makedirs(path + 'labels/static')
  
        full_path = path + "labels/static/" + filename 
        fig.write_image(full_path + '.svg')
        
    if save_html is True:
        if not os.path.exists(path + 'labels'):
            os.makedirs(path + 'labels')
        if not os.path.exists(path + 'labels/dynamic'):
            os.makedirs(path + 'labels/dynamic')
  
        full_path = path + "labels/dynamic/" + filename 
        fig.write_html(full_path + '.html')

    fig.show()
    
    
def plot_thresholds(df, labels, timestamps=None, columns_list=None, 
                                                    save_static=False, save_html=False, 
                                                    path='./reports/figures/', filename='thresholds'):
    
    offset = len(df) - len(labels)  # the offset is the len of the window
    x_len = len(labels)
    if timestamps is None:
        x = np.linspace(0, x_len-1, x_len)
    else:
        x = timestamps[offset:]
    
    start_renderer()
    
    if columns_list is None:
        columns_list = df.columns.to_list()
    
    df = df[columns_list]
    
    rows = int(len(columns_list)/2) if len(columns_list)%2==0 else int(len(columns_list)/2)+1
    cols = 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=columns_list,
        horizontal_spacing = 0.05,
        vertical_spacing = 0.02
    )

    i = 1
    j = 1
    
    for metric in df.columns.values:
        # labels
        fig.add_trace(
            go.Scatter(x=x, y=np.max(df[metric][offset:]) * labels, marker = {'color' : '#00CC96'}, opacity=.6),
            row=i, col=j
        )
        # metric
        fig.add_trace(
            go.Scatter(x=x, y=df[metric][offset:], marker = {'color' : '#636EFA'}),
            row=i, col=j
        )
        # anomalies
        fig.add_trace(
            go.Scatter(x=x, y=df[metric][offset:] * labels, marker = {'color' : '#EF553B'}),
            row=i, col=j
        )
        
        if j >= cols:
            j = 1
            i += 1
        else:
            j += 1

    fig.update_layout(
        height=385*rows, width=600*cols,
        showlegend=False
    )
    
    if save_static is True:
        if not os.path.exists(path + 'thresholds'):
            os.makedirs(path + 'thresholds')
        if not os.path.exists(path + 'thresholds/static'):
            os.makedirs(path + 'thresholds/static')
  
        full_path = path + "thresholds/static/" + filename 
        fig.write_image(full_path + '.svg')
        
    if save_html is True:
        if not os.path.exists(path + 'thresholds'):
            os.makedirs(path + 'thresholds')
        if not os.path.exists(path + 'thresholds/dynamic'):
            os.makedirs(path + 'thresholds/dynamic')
  
        full_path = path + "thresholds/dynamic/" + filename 
        fig.write_html(full_path + '.html')

    fig.show()




