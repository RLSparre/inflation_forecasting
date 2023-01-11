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

class one_layer_LSTM(dl_model):

    def __init__(self, first_layer_neurons=64, learning_rate=0.001, epochs=5, batch_size=1):
        super().__init__()
        self.first_layer_neurons = first_layer_neurons,
        self.learning_rate = learning_rate,
        self.epochs = epochs,
        self.batch_size = batch_size

    def fit_model(self):
        # model
        model = Sequential()
        model.add(LSTM(selffirst_layer_neurons, input_shape=(self.window_size, self.n_features)))
        model.add(Dense(1, activation='linear'))

        # optimizer, loss, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = 'mean_squared_error'
        metrics = 'mean_absolute_error'

        # callbacks
        earlystop = EarlyStopping(monitor='val_loss',
                                  patience=50,
                                  mode='min',
                                  verbose=0)

         # compile and fit
        model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics)
        fit = model.fit(self.X_train,
                        self.y_train,
                        validation_data=(self.X_val, self.y_val),
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        callbacks=[earlystop],
                        verbose=0)


        # print model metrics
        model_metrics = self._model_metrics(model)
        self.model_metrics = model_metrics

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
