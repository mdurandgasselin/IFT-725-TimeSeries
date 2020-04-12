# IFT 725 TimeSeries

Session project in our Deep Learning course at the University of Sherbrooke. It focuses on the study of the effectiveness of deep learning methods of time series forecasting.

## Requirements

The project was developed on python 3.6 on Ubuntu 18.04. So for running the project make sure you have a python version 3.x and it is better to run it on linux.

## Running

* if you want to try ARIMA or LSTM then do :

```
$ jupyter notebook
```

and go to the "execution" directory to see arima.ipynb or lstm.ipynb

* if you want to try SeriesNet you have to do :

For training

```
$ python train.py --company=AAPL_data --num_epochs=7 --pts_2_pred=20 --nb_causal_blk=6
```

For hyperparameter search

```
$ python hyperparam_search.py --company=AAPL_data --nb_causal_blk 5 6 7  --nb_filter 16 24 32  --nb_drop_blk 3 4  --lr 0.001 0.0005
```

## Writing

We also write a report that comes with the code.
