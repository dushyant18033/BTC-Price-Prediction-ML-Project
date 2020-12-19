# BTC-Price-Prediction-ML-Project

## About
This project focuses on the prediction of the prices of Bitcoin, the most in-demand crypto-currency of today’s world. We predict the prices accurately by gathering data available at coinmarketcap while taking various hyper-parameters into consideration which have affected the bitcoin prices until now. 

### ML Models Used:
* Regression Models
  * Linear Regression with various penalties
  * Polynomial Regression
  * Bayesian Regression 
* ARIMA Models
  * AR
  * ARMA
  * ARIMA
  * SARIMAX
* ARIMA + GARCH Models
  * //ishan add

### Python Dependencies:
* pandas
* numpy
* requests
* matplotlib
* statsmodels
* pmdarima

### How to run (Windows/Linux/Mac):
1. cd \<PROJECT ROOT DIRECTORY\>
2. pip install -r requirements.txt
3. python <filename>.py


### File Descriptions:
* auto-ARIMA.py: Runs automated gridsearch from pmdarima library, to find the best model parameters.
* AR.py, ARMA.py, ARIMA.py, SARIMAX.py use the above found best parameters to train the respective models as per their filenames.
* //sajag apne add karde//
* //arima+garch//
