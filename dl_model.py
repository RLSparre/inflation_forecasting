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
from sklearn.metrics import mean_squared_error
tf.random.set_seed(1234)
TUNER_SEED = 5678
HYPERBAND_MAX_EPOCHS = 10
EPOCHS_SEARCH = 50
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

class DeepLearningModel:

    def __init__(self, path, X_train, y_train, X_val, y_val, X_test, y_test):
        self.path = path
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


    def hypermodel(self, deep_learning_model='one_layer_lstm'):
        '''
        This function runs hyperparameter tuning using keras-tuner to find the best model. It then retrains the best
        model and creates outputs.

        :param folder: folder in which figures outputs are saved
        :return:
        '''
        if deep_learning_model == 'one_layer_ff':
            dl_model = self._build_one_layer_ff
            folder = 'one_layer_ff/'
        elif deep_learning_model == 'multi_layer_ff':
            dl_model = self._build_multi_layer_ff
            folder = 'multi_layer_ff/'
        elif deep_learning_model == 'one_layer_lstm':
            dl_model = self._build_one_layer_lstm
            folder = 'one_layer_lstm/'
        elif deep_learning_model == 'multi_layer_lstm':
            dl_model = self._build_multi_layer_lstm
            folder = 'multi_layer_lstm/'
        elif deep_learning_model == 'one_layer_cnn':
            dl_model = self._build_one_layer_cnn
            folder = 'one_layer_cnn/'
        elif deep_learning_model == 'multi_layer_cnn':
            dl_model = self._build_multi_layer_cnn
            folder = 'multi_layer_cnn/'
        else:
            return 'Choose one of the following models: one_layer_lstm, two_layer_lstm_with_dropout'


        # initate keras-tuner
        tuner = kt.Hyperband(dl_model,
                             objective='val_loss',
                             max_epochs=HYPERBAND_MAX_EPOCHS,
                             factor=3,
                             seed=TUNER_SEED,
                             directory=self.path+folder,
                             project_name='model_tuning')

        # create callback
        stop_early = EarlyStopping(monitor="val_loss", patience=10, verbose=0)

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
                            epochs=50,
                            validation_data=(self.X_val, self.y_val))

        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
        # plot figure of history
        utils.plot_fit(history, self.path, folder, best_epoch, deep_learning_model, best_hps.values)

        # Re-instantiate the model and train it with the optimal number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)
        fit = hypermodel.fit(self.X_train,
                             self.y_train,
                             epochs=best_epoch,
                             validation_data=(self.X_val, self.y_val))


        # evaluate model
        model_metrics = self._model_metrics(model, folder, deep_learning_model)

        return model, model_metrics


    def _build_one_layer_lstm(self, hp):
        '''
        Supplementary function that builds a one layer LSTM model with hyperparameters set up for tuning. The function
        is passed to the keras-tuner in the main function 'one_layer_LSTM' for optimization.

        :param hp: hyperparameters
        :return:
        '''

        # initiate model
        model = Sequential()

        # linear space for values of neurons in LSTM layer
        # add LSTM layer
        model.add(LSTM(units=hp.Int('conv_filter', min_value=16, max_value=256, step=16),
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        # add output layer
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value = 1e-5,
                                    max_value = 1e-2,
                                    sampling='LOG',
                                    default = 1e-3)

        # optimizer, loss, metrics
        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_multi_layer_lstm(self, hp):

        model = Sequential()
        # linear space for values of neurons in LSTM layer
        model.add(LSTM(hp.Int('units_1', min_value=16, max_value=256, step=16),
                       return_sequences=True,
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(hp.Int('units_2', min_value=16, max_value=256, step=16)))
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

        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_one_layer_cnn(self, hp):
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

        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_multi_layer_cnn(self, hp):
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

        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_one_layer_ff(self, hp):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(hp.Int('units', min_value=16, max_value=256, step=16), activation='relu'))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        hp_learning_rate = hp.Float('learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-2,
                                    sampling='LOG',
                                    default=1e-3)

        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _build_multi_layer_ff(self, hp):
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

        self._compile_and_fit(model, hp_learning_rate)

        return model

    def _compile_and_fit(self, model, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)


    def _model_metrics(self, model, folder, model_name):
        train_pred = model.predict(self.X_train).flatten()
        val_pred = model.predict(self.X_val).flatten()
        test_pred = model.predict(self.X_test).flatten()

        train_score = mean_squared_error(self.y_train, train_pred)
        test_score = mean_squared_error(self.y_test, test_pred)
        model_error = self.y_test - test_pred

        print('Train Score: %.2f MSE ' % (train_score))
        print('Test Score: %.2f MSE ' % (test_score))
        print('Mean Model Error: %.6f' % (model_error.mean()))

        predictions = np.concatenate((train_pred, val_pred, test_pred))
        self.__plot_predictions_vs_actual(predictions, model_name, folder)

        return {'train_score': train_score,
                'test_score': test_score,
                "mean_model_error": model_error.mean()}

    # TODO: function to plot predictions vs. actual
    def __plot_predictions_vs_actual(self, predictions, model_name, folder):
        # date of first observation (day set to 31st to circumvent some pandas bug)
        first_date_obs = pd.Timestamp('1959-01-31') + pd.DateOffset(months=self.X_train.shape[1])
        obs_index = pd.date_range(first_date_obs, periods=len(predictions), freq='M') - pd.offsets.MonthBegin(1)
        actuals = np.concatenate((self.y_train, self.y_val, self.y_test))


        fig = plt.figure(figsize=(16, 6))
        plt.title(f"{model_name}: Predicted vs. Actual Values")
        plt.plot(obs_index, actuals, label='Actual')
        plt.plot(obs_index, predictions, label='Prediction')
        plt.axvline(obs_index[self.X_train.shape[0]], ls='--', c='black', label='Train/Val Split')
        plt.axvline(obs_index[self.X_train.shape[0]+self.X_val.shape[0]], ls='--', c='red', label='Val/Test Split')
        plt.legend()

        fig.savefig(self.path + folder + 'figures/predictions_vs_actuals', dpi=fig.dpi)