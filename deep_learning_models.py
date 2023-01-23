import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
tf.random.set_seed(1234)
TUNER_SEED = 5678
HYPERBAND_MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
EPOCHS_SEARCH = 50
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
HYPERMODEL_EPOCHS = 50
SUBSEQUENCES = 2
PROJECT_NAME = 'MinMaxScaler_zero_to_one'

class DeepLearningModel:

    def __init__(self,
                 path=os.getcwd(),
                 steps_ahead=1,
                 feature_range=(0,1),
                 window_size=12):
        '''
        :path:
        :steps_ahead:       Forecast horizon. Default 1 month.
        :feature_range:     MinMaxScaler range. Default 0 to 1.
        :window_size:       Number of time steps in each sample. Default 12, i.e. 1 year
        '''

        self.path = path
        self.window_size = window_size
        # get data
        self.df, self.target, self.oos_sample = utils.get_data(window_size=self.window_size)
        # transform target to percentage points (purely for cosmetics)
        self.target = self.target[self.df.index[0]:self.df.index[-1] + pd.DateOffset(months=steps_ahead)] * 100
        # initiate scaler and fit
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.features = self.scaler.fit_transform(self.df)
        # generate samples of #samples x #window_size (timesteps) x #features
        self.X_sample, self.y_sample = utils.samples(features=self.features,
                                                     target=self.target,
                                                     window_size=self.window_size)
        # create train, val and test set
        self.train_len = len(self.X_sample) - 24
        self.val_len = len(self.X_sample) - 12
        self.X_train, self.y_train = self.X_sample[:self.train_len], self.y_sample[:self.train_len]
        self.X_val, self.y_val = self.X_sample[self.train_len:self.val_len], self.y_sample[self.train_len:self.val_len]
        self.X_test, self.y_test = self.X_sample[self.val_len:],self.y_sample[self.val_len:]
        # scale oos sample
        self.oos_sample = self.scaler.transform(self.oos_sample)
        # reshape oos sample to #samples x #window_size (timesteps) x #features
        self.oos_sample = np.array(self.oos_sample).reshape((1,
                                                             self.oos_sample.shape[0],
                                                             self.oos_sample.shape[1]))
        # save index for plotting
        self.index = self.df.index[self.window_size-1:]+pd.DateOffset(months=steps_ahead)

    def hypermodel(self, deep_learning_model):
        '''
        This function runs hyperparameter tuning using keras-tuner to find the best model. It then retrains the best
        model and creates outputs.

        :deep_learning_model:   Which model to train.
        '''

        if deep_learning_model == 'one_layer_ff':
            dl_model = self._build_one_layer_ff
            folder = '/one_layer_ff/'
        elif deep_learning_model == 'multi_layer_ff':
            dl_model = self._build_multi_layer_ff
            folder = '/multi_layer_ff/'
        elif deep_learning_model == 'one_layer_lstm':
            dl_model = self._build_one_layer_lstm
            folder = '/one_layer_lstm/'
        elif deep_learning_model == 'one_layer_gru':
            dl_model = self._build_one_layer_gru
            folder = '/one_layer_gru/'
        elif deep_learning_model == 'multi_layer_gru':
            dl_model = self._build_multi_layer_gru
            folder = '/multi_layer_gru/'
        elif deep_learning_model == 'multi_layer_lstm':
            dl_model = self._build_multi_layer_lstm
            folder = '/multi_layer_lstm/'
        elif deep_learning_model == 'one_layer_cnn':
            dl_model = self._build_one_layer_cnn
            folder = '/one_layer_cnn/'
        elif deep_learning_model == 'multi_layer_cnn':
            dl_model = self._build_multi_layer_cnn
            folder = '/multi_layer_cnn/'
        elif deep_learning_model == 'cnn_lstm':
            # the samples must be reshaped to #samples x #subsequences x #window_size (time steps) x #features
            subsequences = SUBSEQUENCES
            timesteps = self.X_train.shape[1] // subsequences
            self.X_train = self.X_train.reshape((self.X_train.shape[0],
                                                 subsequences,
                                                 timesteps,
                                                 self.X_train.shape[2]))
            self.X_val = self.X_val.reshape((self.X_val.shape[0],
                                                 subsequences,
                                                 timesteps,
                                                 self.X_val.shape[2]))
            self.X_test = self.X_test.reshape((self.X_test.shape[0],
                                                 subsequences,
                                                 timesteps,
                                                 self.X_test.shape[2]))
            self.oos_sample = self.oos_sample.reshape((self.oos_sample.shape[0],
                                                       subsequences,
                                                       timesteps,
                                                       self.oos_sample.shape[2]))
            dl_model = self._build_cnn_lstm
            folder = '/cnn_lstm/'
        else:
            return 'Choose valid model'


        # initate keras-tuner
        tuner = kt.Hyperband(dl_model,
                             objective='val_loss',
                             max_epochs=HYPERBAND_MAX_EPOCHS,
                             factor=3,
                             seed=TUNER_SEED,
                             directory=os.path.join(self.path, folder),
                             project_name=f"{PROJECT_NAME}_keras_tuner")

        # create callback
        stop_early = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, verbose=0)

        # search for optimal hyperparameters using keras-tuner
        tuner.search(self.X_train,
                     self.y_train,
                     epochs=EPOCHS_SEARCH,
                     validation_data=(self.X_val, self.y_val),
                     callbacks=[stop_early])

        # best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"The best hyperparameters for {deep_learning_model} model are: \n {best_hps.values}")


        # build the model with the optimal hyperparameters to find optimal number of epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.X_train,
                            self.y_train,
                            epochs=HYPERMODEL_EPOCHS,
                            validation_data=(self.X_val, self.y_val))

        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
        # plot figure of history
        utils.plot_fit(history,
                       f"{PROJECT_NAME}_fit_figure",
                       self.path,
                       folder,
                       best_epoch,
                       deep_learning_model,
                       best_hps.values)

        # Re-instantiate the model and train it with the optimal number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)
        fit = hypermodel.fit(self.X_train,
                             self.y_train,
                             epochs=best_epoch,
                             validation_data=(self.X_val, self.y_val))


        # evaluate model
        model_metrics = self._model_metrics(model, folder, deep_learning_model)

        # oos prediction
        oos_sample_pred = model.predict(self.oos_sample).flatten()

        return model, model_metrics, oos_sample_pred


    def _build_one_layer_lstm(self, hp):
        '''
        Supplementary function that builds a one layer LSTM model with hyperparameters set up for tuning. The function
        is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''

        # initiate model
        model = Sequential()

        # linear space for values of neurons in LSTM layer
        # add LSTM layer
        model.add(LSTM(hp.Int('units_1', min_value=16, max_value=256, step=16),
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                       activation='relu'))
        # add output layer
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value = 1e-5,
                                    max_value = 1e-2,
                                    sampling='LOG',
                                    default = 1e-3)

        # compile and fit model
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_multi_layer_lstm(self, hp):
        '''
        Supplementary function that builds a multi layer LSTM model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''


        model = Sequential()
        # linear space for values of neurons in LSTM layer
        model.add(LSTM(hp.Int('units_1', min_value=16, max_value=256, step=16),
                       return_sequences=True,
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                       activation='relu'))
        model.add(LSTM(hp.Int('units_2', min_value=16, max_value=256, step=16),
                  activation='relu'))
        # linear space for values of dropout rates in dropout layer
        model.add(Dropout(hp.Float('dropout',
                                   min_value=0.0,
                                   max_value=0.5,
                                   default=0.25,
                                   step=0.05)))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit model
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_one_layer_cnn(self, hp):
        '''
        Supplementary function that builds a one layer CNN model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''
        model = Sequential()

        model.add(Conv1D(filters=hp.Int('conv_filter', min_value=16, max_value=256, step=16),
                         kernel_size=CNN_KERNEL_SIZE,
                         padding='same',
                         input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=CNN_POOL_SIZE))
        model.add(Flatten())
        model.add(Dense(hp.Int('units', min_value=16, max_value=256, step=16),
                        activation='relu'))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit model
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_multi_layer_cnn(self, hp):
        '''
        Supplementary function that builds a multi layer CNN model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''

        model = Sequential()
        # first CNN layer
        model.add(Conv1D(filters=hp.Int('conv_filter_1', min_value=16, max_value=256, step=16),
                         kernel_size=CNN_KERNEL_SIZE,
                         padding='same',
                         input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=CNN_POOL_SIZE))

        # second CNN layer
        model.add(Conv1D(filters=hp.Int('conv_filter_2', min_value=16, max_value=256, step=16),
                         kernel_size=CNN_KERNEL_SIZE,
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=CNN_POOL_SIZE))
        model.add(Flatten())

        # dropout layer
        model.add(Dropout(hp.Float('dropout',
                                   min_value=0.0,
                                   max_value=0.5,
                                   default=0.25,
                                   step=0.05)))

        # fully connected layer
        model.add(Dense(hp.Int('units', min_value=16, max_value=256, step=16),
                        activation='relu'))

        #output layer
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit model
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_one_layer_ff(self, hp):
        '''
        Supplementary function that builds a one layer FF model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(hp.Int('units', min_value=16, max_value=256, step=16),
                        activation='relu'))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit model
        self._compile_and_fit(model, hp_learning_rate)

        return model


    def _build_multi_layer_ff(self, hp):
        '''
        Supplementary function that builds a multi layer FF model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(hp.Int('units_1', min_value=16, max_value=256, step=16),
                        input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                        activation='relu'))
        model.add(Dense(hp.Int('units_2', min_value=16, max_value=256, step=16),
                        activation='relu'))
        model.add(Dropout(hp.Float('dropout',
                                   min_value=0.0,
                                   max_value=0.5,
                                   default=0.25,
                                   step=0.05)))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit model
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_one_layer_gru(self, hp):
        '''
        Supplementary function that builds a one layer GRU model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''

        # initiate model
        model = Sequential()

        model.add(GRU(units=hp.Int('units', min_value=16, max_value=256, step=16),
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                      activation='relu'))
        # add output layer
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_multi_layer_gru(self, hp):
        '''
        Supplementary function that builds a multi layer GRU model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''
        model = Sequential()
        # linear space for values of neurons in GRU layer
        model.add(GRU(hp.Int('units_1', min_value=16, max_value=256, step=16),
                      return_sequences=True,
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                      activation='relu'))
        model.add(GRU(hp.Int('units_2', min_value=16, max_value=256, step=16),
                  activation='relu'))
        # linear space for values of dropout rates in dropout layer
        model.add(Dropout(hp.Float('dropout',
                                   min_value=0.0,
                                   max_value=0.5,
                                   default=0.25,
                                   step=0.05)))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        # compile and fit
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_cnn_lstm(self, hp):
        '''
        Supplementary function that builds a CNN-LSTM model with hyperparameters set up for tuning.
        The function is passed to the keras-tuner in the function 'hypermodel' for optimization.

        :param hp: hyperparameters
        '''

        model = Sequential()
        # timedistributed layer
        model.add(TimeDistributed(Conv1D(
            filters=hp.Int('conv_filter', min_value=16, max_value=256, step=16),
            kernel_size=CNN_KERNEL_SIZE,
            padding='same',
            input_shape=(None, self.X_train.shape[2], self.X_train.shape[3]),
            activation='relu'
        )))
        model.add(TimeDistributed(MaxPooling1D(pool_size=CNN_POOL_SIZE)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(hp.Int('units', min_value=16, max_value=256, step=16),
                      activation='relu'))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        self._compile_and_fit(model, hp_learning_rate)

        return model
    def _compile_and_fit(self, model, learning_rate):
        '''
        This function compiles and fits a given model for a given learning rate with ADAM optimization for MSE loss.

        :model:
        :learning_rate:
        '''

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)


    def _model_metrics(self, model, folder, model_name):
        '''
        This function computes model metrics (MSE for each sample set and MAE for test set) for a given model.
        Additionally, the function calls '__plot_predictions_vs_actuals', which creates a plot for the predictions
        and adds the model metrics to the plot

        :model:
        :folder:        Folder to save figure.
        :model_name:    Model name to add to figure.
        '''
        train_pred = model.predict(self.X_train).flatten()
        val_pred = model.predict(self.X_val).flatten()
        test_pred = model.predict(self.X_test).flatten()

        train_score = mean_squared_error(self.y_train, train_pred)
        val_score = mean_squared_error(self.y_val, val_pred)
        test_score = mean_squared_error(self.y_test, test_pred)
        model_abs_error = np.abs(self.y_test - test_pred)

        print('Train Score: %.2f MSE ' % (train_score))
        print('Validation Score: %.2f MSE ' % (val_score))
        print('Test Score: %.2f MSE ' % (test_score))
        print('Mean Absolute Error: %.6f' % (model_abs_error.mean()))

        model_metrics = {'Train Score': round(train_score,4),
                'Validation Score': round(val_score,4),
                'Test Score': round(test_score,4),
                "Mean Absolute Error": round(model_abs_error.mean(),4)}

        predictions = np.concatenate((train_pred, val_pred, test_pred))
        self.__plot_predictions_vs_actual(predictions, model_name, model_metrics, folder)

        return model_metrics


    def __plot_predictions_vs_actual(self, predictions, model_name, model_metrics, folder):
        '''
        This is an auxiliary function called in '_model_metrics', which plots the predictions versus the actual values.

        :predictions:       Model predictions.
        :model_name:        Model name.
        :model_metrics:     Model metrics.
        :folder:            Folder to save figure.
        '''
        actuals = np.concatenate((self.y_train, self.y_val, self.y_test))

        fig = plt.figure(figsize=(16, 6))
        plt.title(f"{model_name}: Predicted vs. Actual Values")
        plt.plot(self.index, actuals, label='Actual')
        plt.plot(self.index, predictions, label='Prediction')
        plt.axvline(self.index[self.train_len], ls='--', c='black', label='Train/Val Split')
        plt.axvline(self.index[self.val_len], ls='--', c='red', label='Val/Test Split')
        plt.figtext(0.5, 0.01, f"Model Metrics: {model_metrics}", ha="center", fontsize=10)
        plt.legend()

        fig.savefig(os.path.join(self.path, folder) + f"/figures/{PROJECT_NAME}_predictions_vs_actuals", dpi=fig.dpi)
