import utils
import os
import tensorflow as tf
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
        self.n_features = X_train.shape[2]
        self.window_size = X_train.shape[1]

    def one_layer_LSTM(self,
                       first_layer_neurons=64,
                       learning_rate=0.001,
                       epochs=100,
                       batch_size=1,
                       folder='one_layer_LSTM/'):
        '''
        This function builds a deep learning model with 1-layer LSTM architecture and fits it

        :param first_layer_neurons:
        :param learning_rate:
        :param epochs:
        :param batch_size:
        :param folder: c
        :return:
        '''

        # model
        model = Sequential()
        model.add(LSTM(first_layer_neurons, input_shape=(self.window_size, self.n_features)))
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
        model.add(LSTM(first_layer_neurons, return_sequences=True, input_shape=(self.window_size, self.n_features)))
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

    def two_layer_LSTM_with_dropout(self,
                                    first_layer_neurons=64,
                                    second_layer_neurons=128,
                                    dropout_prob=0.2,
                                    learning_rate=0.001,
                                    epochs=100,
                                    batch_size=1,
                                    folder='two_layer_LSTM_with_dropout/'):
        # model
        model = Sequential()
        model.add(LSTM(first_layer_neurons, return_sequences=True, input_shape=(self.window_size, self.n_features)))
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
                        verbose=0)

        model = model

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
                         input_shape=(self.window_size, self.n_features)))
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
                         input_shape=(self.window_size, self.n_features)))
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
