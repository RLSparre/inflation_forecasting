# inflation_forecasting
This is a repository built for investigating and exploring the possibilities of deep learning models to predict
macroeconomic variables in a data rich environment.
More specifically, the aim is to predict the next month YoY CPI number using the FREDMD data set, which consists of
a large number of macroeconomic variables reported on a monthly basis. 

The starting point for the analysis is simple one layer models for a Feed Forward Neural Network (FFNN), 
a Convolution Neural Network (CNN), and the following Recurrent Neural Network (RNN) models:
Gated Recurrent Unit (GRU) and Long-Short-Term-Memory (LSTM). Second, multi layer models are considered; in this case,
two layers of the respective model architecture and a Dropout Layer. Finally, a hybrid model architecture is considered
with a CNN layer for the input followed by a LSTM layer. 
