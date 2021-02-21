import pandas as pd

import pandas as pd
import numpy as np,sys
import tensorflow as tf
import matplotlib.pyplot as plt
from dateutil.parser import parse

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from numpy import newaxis

# 0. Get the Data and simple sorting and check NaN
df = pd.read_csv('./OHLCVs.csv',delimiter=',',usecols=['Unnamed: 0','open','high','low','close', 'pair'])
df = df[df["pair"]=="btc-usd"]
df.drop(["pair"], axis=1)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Pair']
df.Date = pd.to_datetime(df.Date)
df['Mean'] = (df.High + df.Low )/2.0
df = df.sort_values(by="Date")

df = df[:100000]




from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df.Mean.values, freq=1) 
trace1 = go.Scatter(
    x = df.Date,y = decomposition.trend,
    name = 'Trend',mode='lines'
)
trace2 = go.Scatter(
    x = df.Date,y = decomposition.seasonal,
    name = 'Seasonal',mode='lines'
)
trace3 = go.Scatter(
    x = df.Date,y = decomposition.resid,
    name = 'Residual',mode='lines'
)
trace4 = go.Scatter(
    x = df.Date,y = df.Mean,
    name = 'Mean Stock Value',mode='lines'
)


# a. Standard Average of Window
Mean_list = list(df.Mean)
window_size = 500
N = len(Mean_list)
std_avg_predictions = list(Mean_list[:window_size])
for pred_idx in range(window_size,N):
    std_avg_predictions.append(np.mean(Mean_list[pred_idx-window_size:pred_idx]))

# b. EXP Average of Window
window_size = 500
run_avg_predictions = []
running_mean = 0.0
run_avg_predictions.append(running_mean)
decay = 0.8

for pred_idx in range(1,N):
    running_mean = running_mean*decay + (1.0-decay)*Mean_list[pred_idx-1]
    run_avg_predictions.append(running_mean)

trace5 = go.Scatter(
    x = df.Date,y = std_avg_predictions,
    name = 'Window AVG',mode='lines'
)
trace6 = go.Scatter(
    x = df.Date,y = run_avg_predictions,
    name = 'Moving AVG',mode='lines'
)




from statsmodels.tsa.ar_model import AR
window_size = 50
ar_list = list(Mean_list[:window_size])
for pred_idx in range(window_size,N):

    current_window = Mean_list[pred_idx-window_size:pred_idx]
    model = AR(current_window)
    model_fit = model.fit()
    if pred_idx % 500 == 0: print(pred_idx)
    current_predict = model_fit.predict(49,49)[0]
    ar_list.append(current_predict)

trace7 = go.Scatter(
    x = df.Date,y = ar_list,
    name = 'Auto Regression',mode='lines'
)

data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7]
plot(data)

import matplotlib.pyplot as plt

plt.plot(decomposition.trend)
plt.plot(decomposition.seasonal)
plt.plot(decomposition.resid)
plt.plot(std_avg_predictions)
plt.plot(run_avg_predictions)
plt.show(ar_list)


