# Stock Price Prediction

## AIM

### To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

<h4> Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset. The prediction accuracy of the model will be assessed by comparing its output with the true stock prices from the test dataset.

Dataset: The dataset consists of two CSV files:

trainset.csv: This file contains historical stock price data of Google, which will be used for training the RNN model. It includes features such as the opening price of the stock.

testset.csv: This file contains additional historical stock price data of Google, which will be used for testing the trained RNN model. Similarly, it includes features such as the opening price of the stock.

Both datasets contain multiple columns, but for this task, only the opening price of the stock (referred to as 'Open') will be used as the feature for predicting future stock prices.

The objective is to build a model that can effectively learn from the patterns in the training data to make accurate predictions on the test data. </h4>


## DESIGN STEPS

### STEP 1:

#### Read and preprocess training data, including scaling and sequence creation.

### STEP 2:

#### Initialize a Sequential model and add SimpleRNN and Dense layers.

### STEP 3:

Compile the model with Adam optimizer and mean squared error loss.

### Step 4:

#### Train the model on the prepared training data.

### Step 5:

#### Preprocess test data, predict using the trained model, and visualize the results.

## PROGRAM

```py
Developed by : SANJAY T
Register No : 212222110039
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
```
```py
dataset_train = pd.read_csv('trainset.csv')
```
```py
dataset_train.columns
```
```py
dataset_train.head()
```
```py
train_set = dataset_train.iloc[:,1:2].values
```
```py
train_set.shape
```
```py
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
```
```py
training_set_scaled.shape
```
```py
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
```
```py
X_train.shape
```
```py
length = 60
n_features = 1
```
```py
model = Sequential([layers.SimpleRNN(50,input_shape=(60,1)),
                    layers.Dense(1)])
```
```py
model.compile(optimizer='adam',loss='mse')
```

```py
model.summary()
```
```py
model.fit(X_train1,y_train,epochs=20, batch_size=32)
```
```py
dataset_test = pd.read_csv('testset.csv')
```

```py
test_set = dataset_test.iloc[:,1:2].values
```

```py
test_set.shape
```
```py
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
```
```py
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
```

```py
X_test.shape
```
```py
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
```
```py
print('Name : SANJAY T Reg_No: 212222110039')
plt.plot(np.arange(0,1384),inputs, color='aqua', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='crimson',
		label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

```py
print('Name : SANJAY T Reg_No: 212222110039')
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```


## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/sanjaythiyagarajan/rnn-stock-price-prediction/assets/119409242/4aa4dfc2-3e8a-49ff-bdf3-7f36d07a90ba)

### Mean Square Error

![image](https://github.com/sanjaythiyagarajan/rnn-stock-price-prediction/assets/119409242/5eb9514e-bb01-4cbd-9ba6-35acac1e6c29)


## RESULT

### Thus a Recurrent Neural Network model for stock price prediction is done.
