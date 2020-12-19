import pandas as pd
import requests
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
warnings.filterwarnings("ignore")



# Fetching data from the server
url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
content = requests.get(url=url, params=param).json()
df = pd.json_normalize(content['data']['quotes'])

# Extracting and renaming the important variables
df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
df['Low'] = df['quote.USD.low']
df['High'] = df['quote.USD.high']
df['Open'] = df['quote.USD.open']
df['Close'] = df['quote.USD.close']
df['Volume'] = df['quote.USD.volume']

# Drop original and redundant columns
df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])

# Creating a new feature for better representing day-wise values
df['Mean'] = (df['Low'] + df['High'])/2

# Cleaning the data for any NaN or Null fields
df = df.dropna()

print("Data Looks like")
print(df.head())


# Creating a copy for making small changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']


# Plotting the true values
dataset_for_prediction['Mean'].plot(color='green', figsize=(15,2))
plt.legend(['Next day value', 'Mean'])
plt.title('Tyson Opening Stock Value')


# normalizing the exogeneous variables
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Mean'}, inplace=True)
print("Normalized X")
print(X.head())


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'BTC Price next day'}, inplace= True)
y.index=dataset_for_prediction.index
print("Normalized y")
print(y.head())


# train-test split (cannot shuffle in case of time series)
train_size=int(len(df) *0.9)
test_size = int(len(df)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()


# running auto-arima grid search to find the best model
step_wise=auto_arima(
    train_y,
    exogenous=train_X,
    start_p=1,
    start_q=1,
    max_p=7,
    max_q=7,
    d=1,
    max_d=7,
    trace=True,
    m=12,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# print final results
print(step_wise.summary())