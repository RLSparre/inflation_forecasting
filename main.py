import utils
from dl_model import DeepLearningModel
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)
tf.random.set_seed(123)

if __name__ == '__main__':

    # set directory
    path = 'C:/Users/rasmu/PycharmProjects/inflation_forecasting/'

    # get dataframe
    df, target, oos_sample = utils.get_data()

    # lead target
    steps_ahead = 1
    steaps_ahead_offset = pd.DateOffset(months=steps_ahead)
    target = target[df.index[0]:df.index[-1] + steaps_ahead_offset]*100

    # scale features
    scaler = MinMaxScaler(feature_range=(0,1))
    features = scaler.fit_transform(df)

    # generate train/val/test sets
    window_size = 12
    X_sample, y_sample = utils.samples(features=features, target=target, window_size=window_size)

    train_len = int(len(X_sample)*0.9)
    val_len = int(len(X_sample)*0.95)

    X_train, y_train = X_sample[:train_len], y_sample[:train_len]
    X_val, y_val = X_sample[train_len:val_len], y_sample[train_len:val_len]
    X_test, y_test = X_sample[val_len:], y_sample[val_len:]

    # create model instance
    model_instance = DeepLearningModel(path, X_train, y_train, X_val, y_val, X_test, y_test)
    model_ff, metrics_ff = model_instance.hypermodel('one_layer_lstm')
    #model_lstm, metrics_lstm = model_instance.hypermodel('one_layer_lstm')
    #model_cnn, metrics_cnn = model_instance.hypermodel('one_layer_cnn')
    #model_lstm2, metrics_lstm2 = model_instance.hypermodel('two_layer_lstm_with_dropout')
    #model_lstm2, metrics_lstm2 = model_instance.two_layer_LSTM_with_dropout(learning_rate=0.01, epochs=50)
    #model_cnn, metrics_cnn = model_instance.one_layer_CNN(learning_rate=0.01, epochs=50)

    #print(metrics_ff)
    #print(metrics_lstm)
    #print(metrics_cnn)
    #print(metrics_lstm2)




