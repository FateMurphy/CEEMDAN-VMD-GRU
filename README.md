CEEMDAN-VMD-GRU
===
CEEMDAN-VMD-GRU Forecasting model (a light version of CEEMDAN_LSTM) is a Python project for decomposition-integration forecasting models based on EMD methods and LSTM. It is a relatively imperfect module but beginners can quickly use it to make a decomposition-integration prediction by `CEEMDAN`, Complete Ensemble Empirical Mode Decomposition with Adaptive Noise [(Torres et al. 2011)](https://ieeexplore.ieee.org/abstract/document/5947265/), and `LSTM`, Long Short-Term Memory recurrent neural network [(Hochreiter and Schmidhuber, 1997)](https://ieeexplore.ieee.org/abstract/document/6795963). If you use or refer to the content of this project, please cite paper: [(F. Zhou, Z. Huang, C. Zhang,
Carbon price forecasting based on CEEMDAN and LSTM, Applied Energy, 2022, Volume 311, 118601, ISSN 0306-2619.)](https://doi.org/10.1016/j.apenergy.2022.118601.)

![](https://github.com/FateMurphy/CEEMDAN-VMD-GRU/blob/main/CEEMDAN-VMD-GRU%20predictor%20flowchart.jpg)

## Start to predict
### 0. Load raw data
Change `CODE` and `PATH` to load dataset from a csv file to create `pd.Series` as raw data.
```python
df_raw_data = pd.read_csv(PATH+CODE+'.csv', header=0, parse_dates=['date'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
```
### 1. Just follow the steps without modification
Run the .py file without any modification, but please pay attention to `pip install modules`
```python
if __name__ == '__main__': ...
```

### 2. Plot and Output
All functions' return are `pd.DataFrame`.  

Use `df.plot()` to show the figure.
```python
df_name.plot(figsize=(12,6), title='CEEMDAN-VMD-GRU Predicting Result')
```

Use `pd.DataFrame.to_csv()` to download the result.
```python
pd.DataFrame.to_csv(df_name, PATH+'CEEMDAN_VMD_GRU_predict_output.csv')
```

## Postscript
For more functions, please use CEEMDAN_LSTM. Thanks!  
If you have any questions, please leave your comment or email me.
