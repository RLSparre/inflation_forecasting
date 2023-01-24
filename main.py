from deep_learning_models import DeepLearningModel
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)
tf.random.set_seed(123)

if __name__ == '__main__':

    # set path to current working directory
    path = os.getcwd()

    # create instance of model with feature range from 0 to 1 for MinMaxScaler
    model_instance = DeepLearningModel(feature_range=(0,1))

    # list of all models included in module
    # NOTE: calling CNN-LSTM model reshapes the samples, so if one wishes to call another model after, a new instance
    #       must be created
    models = ['one_layer_ff', 'one_layer_cnn', 'one_layer_gru', 'one_layer_lstm', 'multi_layer_ff', 'multi_layer_cnn',
              'multi_layer_gru', 'multi_layer_lstm', 'cnn_lstm']

    results, oos_sample_preds = [], []

    # iterate through models and store results
    for model in models:
        _, metrics, oos_sample_pred = model_instance.hypermodel(model)
        results.append(list(metrics.values()))
        oos_sample_preds.append(oos_sample_pred)

    # save metrics and predictionsmode
    metrics_df = pd.DataFrame(results, columns=metrics.keys(), index=models)
    metrics_df.to_csv(path+'/zero_to_one_MinMaxScaler_metrics.csv', sep=';', decimal='.')
    prediction_df = pd.DataFrame(oos_sample_preds, columns=['Prediction'], index=models)
    prediction_df.to_csv(path + '/zero_to_one_MinMaxScaler_predictions.csv', sep=';', decimal='.')




