import utils
import os
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error


class dl_model:

    def __init__(self, path, X_train, y_train, X_val, y_val, X_test, y_test):
        self.path = path
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


    def one_layer_LSTM(self, folder='one_layer_LSTM/'):
        '''
        This function runs hyperparameter tuning using keras-tuner to find the best model. It then retrains the best
        model and creates outputs.

        :param folder: folder in which figures outputs are saved
        :return:
        '''

        # initate keras-tuner
        tuner = kt.Hyperband(self._build_one_layer_LSTM,
                             objective='val_loss',
                             max_epochs=10,
                             factor=3)

        # create callback
        stop_early = EarlyStopping(monitor="val_loss", patience=5)

        # search for optimal hyperparameters using keras-tuner
        tuner.search(self.X_train,
                     self.y_train,
                     epochs=50,
                     validation_data=(self.X_val, self.y_val),
                     callbacks=[stop_early])

        # best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first LSTM layer is 
        {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        """)

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
        utils.plot_fit(history, self.path, folder, best_epoch)

        # Re-instantiate the model and train it with the optimal number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)
        fit = hypermodel.fit(self.X_train,
                       self.y_train,
                       epochs=best_epoch,
                       validation_data=(self.X_val, self.y_val))


        # evaluate model
        model_metrics = self._model_metrics(model)

        return model, model_metrics


    def _build_one_layer_LSTM(self, hp):
        '''
        Supplementary function that builds a one layer LSTM model with hyperparameters set up for tuning. The function
        is passed to the keras-tuner in the main function 'one_layer_LSTM' for optimization.

        :param hp: hyperparameters
        :return:
        '''

        # initiate model
        model = Sequential()

        # linear space for values of neurons in LSTM layer
        hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
        # add LSTM layer
        model.add(LSTM(units=hp_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        # add output layer
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        learning_rate_vals = [10**-x for x in np.arange(2.0, 4.5, 0.5)]
        hp_learning_rate = hp.Choice('learning_rate', values=learning_rate_vals)

        # optimizer, loss, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        return model

    # TODO: Implement same logic as for one layer lstm
    def _build_two_layer_with_dropout_LSTM(self, hp):

        model = Sequential()
        # linear space for values of neurons in LSTM layer
        hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
        model.add(LSTM(hp_units,
                       return_sequences=True,
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(hp_units))
        dropout_rate_vals = [0.2, 0.35, 0.5]
        hp_dropout_rate = hp.Choice('dropout_rate', values=dropout_rate_vals)
        model.add(Dropout(hp_dropout_rate))
        model.add(Dense(1, activation='linear'))

        # log space for values of learning rates
        learning_rate_vals = [10 ** -x for x in np.arange(2.0, 4.5, 0.5)]
        hp_learning_rate = hp.Choice('learning_rate', values=learning_rate_vals)

        # optimizer, loss, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        return model

    def two_layer_LSTM_with_dropout(self,
                                    first_layer_neurons=64,
                                    second_layer_neurons=128,
                                    dropout_prob=0.2,
                                    learning_rate=0.001,
                                    epochs=100,
                                    batch_size=1,
                                    folder='two_layer_LSTM_with_dropout/'):
        '''
        This function builds a deep learning model with a 2 layer LSTM and Dropout layer architecture and fits it

        :param first_layer_neurons:
        :param second_layer_neurons:
        :param dropout_prob:
        :param learning_rate:
        :param epochs:
        :param batch_size:
        :param folder: This function builds a deep learning model with 1-layer LSTM architecture and fits it
        :return:
        '''
        # model
        model = Sequential()
        model.add(LSTM(first_layer_neurons,
                       return_sequences=True,
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(second_layer_neurons))
        model.add(Dropout(dropout_prob))
        model.add(Dense(1, activation='linear'))

        # optimizer, loss, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # callbacks
        earlystop = EarlyStopping(monitor='val_loss',
                                  patience=50,
                                  mode='min',
                                  verbose=0)
        checkpoint = ModelCheckpoint(folder, save_best_only=True)

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        fit = model.fit(self.X_train,
                        self.y_train,
                        validation_data=(self.X_val, self.y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earlystop, checkpoint],
                        verbose=2)

        # plot figure of fit
        fit_figure = utils.plot_fit(fit)

        # save fit figure in 'figures' folder (which is created if it does not exist)
        fig_path = self.path+folder+'figures/'
        fig_path_exists = os.path.exists(fig_path)
        if not fig_path_exists:
            os.makedirs(fig_path)
        fit_figure.savefig(fig_path+'fit_figure', dpi=fit_figure.dpi)

        # print model metrics
        model_metrics = self._model_metrics(model)

        return model, model_metrics


    def one_layer_CNN(self,
                      conv_layer_filters=64,
                      kernel_size=2,
                      pool_size=2,
                      dense_layer_neurons=128,
                      learning_rate=0.001,
                      epochs=500,
                      batch_size=1,
                      folder='one_layer_CNN/'):

        # model
        model = Sequential()
        model.add(Conv1D(conv_layer_filters,
                         kernel_size=kernel_size,
                         activation='relu',
                         input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(dense_layer_neurons, activation='relu'))
        model.add(Dense(1, activation='linear'))

        # optimizer, loss, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # callbacks
        earlystop = EarlyStopping(monitor='val_loss',
                                  patience=50,
                                  mode='min',
                                  verbose=0)
        checkpoint = ModelCheckpoint(folder, save_best_only=True)

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        fit = model.fit(self.X_train,
                        self.y_train,
                        validation_data=(self.X_val, self.y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earlystop, checkpoint],
                        verbose=0)

        # plot figure of fit
        fit_figure = utils.plot_fit(fit)

        # save fit figure in 'figures' folder (which is created if it does not exist)
        fig_path = self.path+folder+'figures/'
        fig_path_exists = os.path.exists(fig_path)
        if not fig_path_exists:
            os.makedirs(fig_path)
        fit_figure.savefig(fig_path+'fit_figure', dpi=fit_figure.dpi)

        # print model metrics
        model_metrics = self._model_metrics(model)

        return model, model_metrics

    # TODO: Figure out how to implement model and code with same logic as one layer lstm
    def CNN_LSTM(self,
                 conv_layer_filters=64,
                 kernel_size=2,
                 pool_size=2,
                 lstm_layer_neurons=128,
                 learning_rate=0.001,
                 epochs=500,
                 batch_size=1,
                 folder='CNN_LSTM/'):

        # model
        model = Sequential()
        model.add(Conv1D(conv_layer_filters,
                         kernel_size=kernel_size,
                         activation='relu',
                         input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(LSTM(lstm_layer_neurons, activation='relu'))
        model.add(Dense(1, activation='linear'))

        # optimizer, loss, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # callbacks
        earlystop = EarlyStopping(monitor='val_loss',
                                  patience=50,
                                  mode='min',
                                  verbose=0)
        checkpoint = ModelCheckpoint(folder, save_best_only=True)

        # compile and fit
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        fit = model.fit(self.X_train,
                        self.y_train,
                        validation_data=(self.X_val, self.y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earlystop, checkpoint],
                        verbose=0)

        # plot figure of fit
        fit_figure = utils.plot_fit(fit)

        # save fit figure in 'figures' folder (which is created if it does not exist)
        fig_path = self.path+folder+'figures/'
        fig_path_exists = os.path.exists(fig_path)
        if not fig_path_exists:
            os.makedirs(fig_path)
        fit_figure.savefig(fig_path+'fit_figure', dpi=fit_figure.dpi)

        # print model metrics
        model_metrics = self._model_metrics(model)

        return model, model_metrics

    def _model_metrics(self, model):
        train_pred = model.predict(self.X_train).flatten()
        val_pred = model.predict(self.X_val).flatten()
        test_pred = model.predict(self.X_test).flatten()

        train_score = mean_squared_error(self.y_train, train_pred)
        test_score = mean_squared_error(self.y_test, test_pred)
        model_error = self.y_test - test_pred

        print('Train Score: %.2f MSE ' % (train_score))
        print('Test Score: %.2f MSE ' % (test_score))
        print('Mean Model Error: %.6f' % (model_error.mean()))

        return {'train_score': train_score,
                'test_score': test_score,
                "mean_model_error": model_error.mean()}

    # TODO: function to plot predictions vs. actual
    def _plot_predictions_vs_actual(self):
        return None
