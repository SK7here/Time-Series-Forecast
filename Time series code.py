#Components of Time series:
    #Trend:         Trend is a general direction in which something is developing or changing
    #Seasonality:   Any predictable change or pattern in a time series that recurs or repeats over a specific time period





#LOADING LIBRARIES
import numpy as np # for scientific computing
import pandas as pd # for working with data
import matplotlib.pyplot as plt  # For plotting graphs 
from datetime import datetime    # To access datetime 
from pandas import Series        # To work on series
import warnings # for throwing exceptions
warnings.filterwarnings("ignore")#To ignore warnings
from sklearn.metrics import mean_squared_error #To use MSE error calculation
from math import sqrt #To find sqrt values
import statsmodels.api as sm #To perform AUGMENTED DICKEY-FULLER TEST
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing #To use Simple, Exponential and Holt Smoothing
from statsmodels.tsa.seasonal import seasonal_decompose #To perform seasonal decomposition
from statsmodels.tsa.stattools import acf, pacf #To perform acf and pacf
from statsmodels.tsa.arima_model import ARIMA #To use ARIMA model

#importing datasets
train=pd.read_csv("Train_data.csv") 

#Creating copy of the dataset
train_original=train.copy() 

#Dimensions of data set
print("\nTRAINING SET\n")
print(train.shape)


############################################################################################################################################################


#Datetime used in the above dataset is of OBJECT type.So we need to typecast it to DATETIME type to perform Feature extraction later
    #Actual dataset
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
    #Copy Dataset
train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')

#Splitting the Datetime of each dataset as year, month, day and hour(numerical format)
    #Creating a list of 4 lists
for i in (train, train_original):
    i['year']=i.Datetime.dt.year 
    i['month']=i.Datetime.dt.month 
    i['day']=i.Datetime.dt.day
    i['Hour']=i.Datetime.dt.hour

#Extracting day(index of the day in the week) from the date we have (eg: Monday -> 0)
train['day of week']=train['Datetime'].dt.dayofweek 
temp = train['Datetime']

#Defining the function to find a date is weekend or not
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

#Applying the function to our dataset    
temp2 = train['Datetime'].apply(applyer) 
#Creating a new label of weekend in our dataset
train['weekend']=temp2
print("\nProcessed Training dataset(FIRST 5)\n")
print(train.head(5))

#Plotting graph
    # indexing the train dataset with Datetime to get the time period on the x-axis. 
train.index = train['Datetime']
plt.figure(figsize=(16,8))
#Label for legend
    #If x not specified, it takes index value
plt.plot(train['Count'], label='Passenger Count') 
plt.title('Time Series') 
plt.xlabel("Time(year-month)") 
plt.ylabel("Passenger count")
#Adds the legend at best position
plt.legend(loc='best')
plt.show()


############################################################################################################################################################


#Exploratory Analysis
#first hypothesis-> traffic will increase as the years pass by
temp=train.groupby('year')['Count'].mean()
temp.plot.bar(figsize=(12,5), title= 'Passenger Count(Yearwise)', fontsize=14)
plt.show()


#second hypothesis-> increase in traffic from May to October
temp=train.groupby('month')['Count'].mean()
temp.plot.bar(figsize=(12,5), title= 'Passenger Count(Monthwise)', fontsize=14)
plt.show()
    #Here, thr mean of last 3 months will be the least
    #Because the count of passengers has not been considered for the last 3 months of the year 2014
    #Since, the traffic is increasing exponentially, these 3 months would have had lot more passengers and mean would have increased
#Count of passengers in each month of each year
temp=train.groupby(['year', 'month'])['Count'].mean() 
temp.plot.bar(figsize=(12,5), title= 'Passenger Count(Yearly Monthwise)', fontsize=14)
plt.show()


#Third hypothesis-> traffic will be more during peak hours
temp=train.groupby('Hour')['Count'].mean()
temp.plot.bar(figsize=(12,5), title= 'Passenger Count(Hourwise)', fontsize=14)
plt.show()

#Fourth hypothesis-> traffic will be more on weekdays.
temp=train.groupby('weekend')['Count'].mean()
temp.plot.bar(figsize=(12,5), title= 'Passenger Count(Weekday/Weekend)', fontsize=14)
plt.show()
#Trying to find out which day has more traffic
temp=train.groupby('day of week')['Count'].mean()
temp.plot(figsize=(12,5), title= 'Passenger Count(Daywise)', fontsize=14)
plt.show()


#Dropping ID variable from training set
train=train.drop('ID',1)
#Setting the datetime as index for training set
train.index = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')

# Converting Count value from training dataset based on hours
hourly = train.Count.resample('H').mean()
print("\nConverting Count value from training dataset based on hours(FIRST 24)\n")
print(hourly.head(24))
# Converting Count value from training dataset based on days
daily = train.Count.resample('D').mean()
print("\nConverting Count value from training dataset based on days(FIRST 9)\n")
print(daily.head(9))
# Converting Count value from training dataset based on weeks
weekly = train.Count.resample('W').mean()
print("\nConverting Count value from training dataset based on weeks(FIRST 5)\n")
print(weekly.head(5))
# Converting Count value from training dataset based on months 
monthly = train.Count.resample('M').mean()
print("\nConverting Count value from training dataset based on months(FIRST 5)\n")
print(monthly.head(5))


#Plotting a consolidated graph containing above 4 conversions
    #Dividing the plot into 4 subplots
    #Plotting only the count of passengers from each set
fig, axs = plt.subplots(4,1) 
hourly.plot(figsize=(12,8), title= 'Hourly', fontsize=14, ax=axs[0])
daily.plot(figsize=(12,8), title= 'Daily', fontsize=14, ax=axs[1])
weekly.plot(figsize=(12,8), title= 'Weekly', fontsize=14, ax=axs[2])
monthly.plot(figsize=(12,8), title= 'Monthly', fontsize=14, ax=axs[3])
plt.show()


#Time series is getting stable correspondingly as we aggregate in terms of daily, weekly and monthly basis.
    #We will work on DAILY time series as it is difficult to convert the monthly and weekly predictions to hourly predictions
train = train.resample('D').mean()
print("\nDAILY time series(FIRST 5)\n")
print(train.head(5))


#Splitting training dataset into training and validation dataset
    #Last 3 months are chosen to be used as validation dataset
    #Rest of the data is training dataset
    #.ix is useful for mixed complex type of lables (date is our label in this case)
Train=train.ix['2012-08-25':'2014-06-24']
valid=train.ix['2014-06-25':'2014-09-25']

#Plotting graph to visualize split of train and validation set
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train')
valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')
#Finally showing the plot combines the above 2 plots into 1 plot
plt.show()
    

############################################################################################################################################################


#Different Modelling techniques
                                                                 #1.NAIVE APPROACH
    #we assume that the next expected point is equal to the last observed point.
    #So we can expect a straight horizontal line as the prediction
#Converting train dataset as array
td= np.asarray(Train.Count)
#Copying valid dataset into y_hat
y_hat = valid.copy()
#Adding a new column "naive" in y_hat
    #Assigning last Count value of train dataset to entire "naive" column of y_hat
y_hat['naive'] = td[len(td)-1]
print("\nNaive method - prediction(FIRST 5)\n")
print(y_hat['naive'].head(5))

#Plotting graph
plt.figure(figsize=(12,8)) 
plt.plot(Train.index, Train['Count'], label='Train') 
plt.plot(valid.index,valid['Count'], label='Valid') 
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title("Naive Forecast") 
plt.show()

#Computing root mean squared error
print("\n Root mean squared error for Naive Approach\n")
Naive_rms = sqrt(mean_squared_error(valid.Count, y_hat.naive)) 
print(Naive_rms)


############################################################################################################################################################


                                                         #2.MOVING AVERAGE
    #Predictions are made on the basis of the average of last few points
#Last 10 observations
#Getting mean of last 10 Count values from train dataset
y_hat['moving_avg_ten'] = Train['Count'].rolling(10).mean().iloc[-1]
print("\nMoving Average Forecast using last 10 observations - prediction(FIRST 5)\n")
print(y_hat['moving_avg_ten'].head(5))

#Plotting graph
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['moving_avg_ten'], label='Moving Average')
plt.title('Moving Average Forecast using last 10 observations')
plt.legend(loc='best') 
plt.show() 

#Computing root mean squared error
print("\n Root mean squared error for Moving Average based on last 10 observations\n")
Mv_avg_10_rms = sqrt(mean_squared_error(valid.Count, y_hat.moving_avg_ten)) 
print(Mv_avg_10_rms)



#Last 30 observations
#Getting mean of last 30 Count values from train dataset
y_hat['moving_avg_thirty'] = Train['Count'].rolling(30).mean().iloc[-1]
print("\nMoving Average Forecast using last 30 observations - prediction(FIRST 5)\n")
print(y_hat['moving_avg_thirty'].head(5))

#Plotting graph 
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['moving_avg_thirty'], label='Moving Average') 
plt.legend(loc='best')
plt.title('Moving Average Forecast using last 30 observations')
plt.show() 

#Computing root mean squared error
print("\n Root mean squared error for Moving Average based on last 30 observations\n")
Mv_avg_30_rms = sqrt(mean_squared_error(valid.Count, y_hat.moving_avg_thirty)) 
print(Mv_avg_30_rms)



#Last 50 observations
#Getting mean of last 50 Count values from train dataset
y_hat['moving_avg_fifty'] = Train['Count'].rolling(50).mean().iloc[-1]
print("\nMoving Average Forecast using last 50 observations - prediction(FIRST 5)\n")
print(y_hat['moving_avg_fifty'].head(5))

#Plotting graph
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['moving_avg_fifty'], label='Moving Average') 
plt.legend(loc='best')
plt.title('Moving Average Forecast using last 50 observations')
plt.show() 

#Computing root mean squared error
print("\n Root mean squared error for Moving Average based on last 50 observations\n")
Mv_avg_50_rms = sqrt(mean_squared_error(valid.Count, y_hat.moving_avg_fifty)) 
print(Mv_avg_50_rms)


############################################################################################################################################################


                                                    #3. Simple Exponential Smoothing

    #we assign larger weights to more recent observations than to observations from the distant past.
#Training simple exponent smoothing model
        #Higher the smoothing_level (ALPHA)-> more weight granted to recent observations
        #If optimized is set to True, optimal smoothing_level will automatically be chosen
fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.75,optimized=False)
#Predicting target variable for validation dataset
y_hat['SES'] = fit2.forecast(len(valid))
print("\nSimple Exponential Smoothing - prediction(FIRST 5)\n")
print(y_hat['SES'].head(5))

#Plotting graph
plt.figure(figsize=(16,8)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['SES'], label='SES') 
plt.legend(loc='best')
plt.title('Simple Exponential Smoothing')
plt.show()

#Computing root mean squared error
print("\n Root mean squared error for Simple Exponential Smoothing\n")
SES_rms = sqrt(mean_squared_error(valid.Count, y_hat.SES)) 
print(SES_rms)


############################################################################################################################################################


                                                         #4. Holt’s Linear Trend Model
    #extension of simple exponential smoothing; This method takes into account the trend of the dataset

#Training Holt’s Linear Trend Model
        #Higher the smoothing_level(ALPHA)-> more weight granted to recent observations
        #Higher the smoothing_slope(BETA)-> higher the degree of considering new data trends than old data trends
        #If optimized is set to True, optimal smoothing_level will automatically be chosen
fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.75,smoothing_slope=0.001)
y_hat['Holt_linear'] = fit1.forecast(len(valid))
print("\nHolt’s Linear Trend Model(FIRST 5)\n")
print(y_hat.Holt_linear.head(5))

#Plotting graph
plt.figure(figsize=(16,8)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['Holt_linear'], label='Holt_linear') 
plt.legend(loc='best')
plt.title('Holt’s Linear Trend Model')
plt.show()

#Computing root mean squared error
print("\n Root mean squared error for Holt’s Linear Trend Model\n")
Holt_Linear_rms = sqrt(mean_squared_error(valid.Count, y_hat.Holt_linear)) 
print(Holt_Linear_rms)


############################################################################################################################################################


                                                     #5.Holt winter’s model on daily time series

    #This model takes the seasonality also into account besides level and trend
#Training Holt winter’s model
#There are very subtle changes in trend and seasonality, so we use "ADDITIVE" model
#seasonal_periods is taken as 7-> same pattern repeats evry week (weekdays, weekends pattern)
fit1 = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
#Predicting target variable for validation dataset
y_hat['Holt_Winter'] = fit1.forecast(len(valid))
print("\nHolt winter’s model- prediction(FIRST 5)\n")
print(y_hat['Holt_Winter'].head(5))

#Plotting graph
plt.figure(figsize=(16,8)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['Holt_Winter'], label='Holt_Winter')
plt.title('Holt winter’s model')
plt.legend(loc='best') 
plt.show()

#Computing root mean squared error
print("\n Root mean squared error for Holt winter’s model\n")
Holt_Winter_rms = sqrt(mean_squared_error(valid.Count, y_hat.Holt_Winter)) 
print(Holt_Winter_rms)


############################################################################################################################################################


#Defining function to know about our time series
#GENERAL NULL HYPOTHESIS IS TIME SERIES IS NON-STATIONARY(TIME DEPENDENT STRUCTURE)
    #STATIONARY-> Mean, Variance and covariance should be constant across time
    #Variance-> how much dataset is distributed
    #Covariance-> joint variability between tow different entities(X & Y)

def test_stationarity(timeseries , heading):
    result = sm.tsa.stattools.adfuller(timeseries)
    #More negative the ADF Statistic, more time independent structure and vice-versa
    print("\n\nAUGMENTED DICKEY-FULLER TEST INFERENCES")
    print('ADF Statistic: %f' % result[0])
    #If p-value <= 0.05, we have time independent structure and vice-versa
    print('p-value: %.20f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    #Visualizing stationarity
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=24).mean() # 24 hours on each day
    #Standard deviation is square root of variance
    rolstd = timeseries.rolling(window=24).std() # 24 hours on each day
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation of {}'.format(heading))
    plt.show()

    
print("\nPerforming AUGMENTED DICKEY-FULLER TEST on training data\n")
test_stationarity(train_original['Count'], 'Training dataset') 
print("\nFrom graph we infer that, still there is a slight increasing trend in the data\n")
print("\nFrom this we observe that our time series is time independent structure\n")
print("ADF Statistic is lesser than all critical values")
print("So,we can reject the NULL HYPOTHESIS\n")
print("We come to a conclusion that we have a TIME INDEPENDENT structure\n")

############################################################################################################################################################

#We can make data more stationary by removing trend and seasonality
    #Taking natural log of count values-> reduces larger numbers greatly than smaller numbers
Train_log = np.log(Train['Count']) 
valid_log = np.log(valid['Count'])



#Methods for removing trend from time series
#1.Using MOVING AVERAGE
    #computes moving average by taking average for every 24 values correspondingly
moving_avg = Train_log.rolling(window=24).mean()

#Plotting graph
plt.plot(Train_log, color='blue', label='Train_log') 
plt.plot(moving_avg, color = 'red', label='Moving_Average')
plt.title('Removing trend- Moving Average')
plt.show()

#Removing Increasing trend by subtracting corresponding mean
train_log_moving_avg_diff = Train_log - moving_avg
#Since we took the average of 24 values, rolling mean is not defined for the first 23 values.
#Dropping null values
train_log_moving_avg_diff.dropna(inplace = True)

print("\n\nREMOVING TREND\n\n")
print("\n1.Moving Average method")
print("\nPerforming AUGMENTED DICKEY-FULLER TEST on trend removed data\n")
test_stationarity(train_log_moving_avg_diff, 'Removing trend- Moving Average')
print("\nWe can see that the Test Statistic is very smaller as compared to the Critical Value\n\n")



#2.DIFFERENCING - Stabilizing Mean
    #Shifting index by one position-> subtracting 2 consecutive values iteratively
print("\n2.DIFFERENCING - Stabilizing Mean")
train_log_diff = Train_log - Train_log.shift(1)
print("\nPerforming AUGMENTED DICKEY-FULLER TEST on trend removed data\n")
test_stationarity(train_log_diff.dropna(), 'Removing trend- Differencing')
print("\nWe can see that the Test Statistic is the smalleat amongst the values we've got")
print("\nTrend is removed to maximum extent")

############################################################################################################################################################

#Removing seasonality from time series
    #use seasonal decompose to extract only residuals, neglecting trend and seasonality
#fequency = 24-> data collected hour by hour
decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, freq = 24) 
#Plotting the seasonal decomposition
trend = decomposition.trend 
seasonal = decomposition.seasonal 
residual = decomposition.resid
plt.subplot(411)
plt.plot(Train_log, label='Original') 
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals') 
plt.legend(loc='best') 
plt.tight_layout()
plt.show()

#Checking stationarity of RESIDUALS-> remains after removing trend and seasonality from time series
train_log_decompose = pd.DataFrame(residual)
#Adding index to residuals as 'date'
train_log_decompose['date'] = Train_log.index 
train_log_decompose.set_index('date', inplace = True)
#Removing null values
train_log_decompose.dropna(inplace=True)
#We have to access in terms of index-> all values in index 0
test_stationarity(train_log_decompose[0], 'Residual')
print("\nIt can be interpreted from the results that the residuals are stationary\n")

############################################################################################################################################################

#We removed trend and seasonality from time series to forecast time series using AUTOREGRSSIVE INTEGRATED MOVING AVERAGE model(ARIMA) 
                                        #7.Forecasting the time series using ARIMA (works only with stationary time series)

#We need to find optimal values for p,d,q parametes of ARIMA model
    #To find that, we use Autocorrelation and partial Autocorrelation functions
        #ACF is a measure of the correlation between the TimeSeries with a lagged version of itself.
            #lags-> previous/past observations
            #For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with series at instant ‘t1-5’…’t2-5’
        #PACF is same as ACF but eliminates the variations already explained by the intervening comparisons
            # Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.
#nlags= n largest lags for which acf is returned
lag_acf = acf(train_log_diff.dropna(), nlags=25)
#nlags= n largest lags for which pacf is returned
#ols method-> ordinary least square regression method-> closely fits function to data by minimizing the sum of squared errors 
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')



#ACF and PACF plot
plt.title('Autocorrelation Function') 
plt.axhline(y=0,linestyle='--',color='gray')
#Plotting upper and lower confidence levels
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.plot(lag_acf)
plt.show()

plt.title('Partial Autocorrelation Function') 
plt.axhline(y=0,linestyle='--',color='gray')
#Plotting upper and lower confidence levels
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.plot(lag_pacf)
plt.show()

print("\n\n\np value is the lag value where the PACF chart crosses the upper confidence interval for the first time")
print("In this case, from graph we infer, p=1")
print("\nq value is the lag value where the ACF chart crosses the upper confidence interval for the first time")
print("In this case, from graph we infer, q=1")


#p: The number of lag observations included in the model, also called the lag order.
#d: The number of times that the raw observations are differenced, also called the degree of differencing.
#q: The size of the moving average window, also called the order of moving average.





#Building AR model
model = ARIMA(Train_log, order=(1, 1, 0))  # here the q value is zero since it is just the AR model 
results_AR = model.fit()
plt.title('AR model')
plt.plot(train_log_diff.dropna(), label='original')
#fittedvalues-> predicted values using the model
plt.plot(results_AR.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best') 
plt.show()

#Predicting Validation data using AR model
AR_predict=results_AR.predict(start="2014-06-25", end="2014-09-25")
print("\n\nPredicting Validation data using AR model")
print(AR_predict.head(5))


#Finding cumulative sum of array elements at each index and assigning that as value for succeeding index
AR_predict=AR_predict.cumsum().shift().fillna(0)
print("\nAfter assiging cumulative sum of preceeding index value as the value for each index")
print(AR_predict)
#Since first validation data is "ZERO", we add first valid data to all entries to compensate
    #Creating a series of first valid data filled equal to no of rows times of valid dataset
AR_predict1=pd.Series( (np.ones(valid.shape[0]) * np.log(valid['Count'])[0]) , index = valid.index)
#Adding 2 series
AR_predict1=AR_predict1.add(AR_predict,fill_value=0)
#Calculating expoenential- to compensate log of each value
AR_predict = np.exp(AR_predict1)

#Plottig AR graph
fig,ax = plt.subplots()
ax.plot(valid['Count'], label = "Valid") 
ax.plot(AR_predict, color = 'red', label = "Predict")
ax.legend(loc= 'best')
#To avoid overlapping of xticks
fig.autofmt_xdate()
plt.title('AR model predictions')
plt.show()

#Computing root mean squared error
    #Since pedictions is in form of series, RMSE cannot be applied in direct form
print("\n Root mean squared error for AR model\n")
#Dividing sqrt of dot product by no of observations
AR_rms = np.sqrt(np.dot(AR_predict, valid['Count']))/valid.shape[0]
print(AR_rms)





#Building MA model
model = ARIMA(Train_log, order=(0, 1, 1))  # here the p value is zero since it is just the MA model 
results_MA = model.fit()  
plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_MA.fittedvalues, color='red', label='prediction') 
plt.legend(loc='best')
plt.title('MA model')
plt.show()

#Predicting Validation data using MA model
MA_predict=results_MA.predict(start="2014-06-25", end="2014-09-25")
print("\n\nPredicting Validation data using MA model")
print(MA_predict.head(5))

#Finding cumulative sum of array elements at each index and assigning that as value for succeeding index
MA_predict=MA_predict.cumsum().shift().fillna(0)
print("\nAfter assiging cumulative sum of preceeding index value as the value for each index")
print(MA_predict)
#Since first validation data is "ZERO", we add first valid data to all entries to compensate
    #Creating a series of first valid data filled equal to no of rows times of valid dataset
MA_predict1=pd.Series( (np.ones(valid.shape[0]) * np.log(valid['Count'])[0]) , index = valid.index)
#Adding 2 series
MA_predict1=MA_predict1.add(MA_predict,fill_value=0)
#Calculating expoenential- to compensate log of each value
MA_predict = np.exp(MA_predict1)

#Plottig MA graph
fig,ax = plt.subplots()
plt.plot(valid['Count'], label = "Valid") 
plt.plot(MA_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best')
#To avoid overlapping of xticks
fig.autofmt_xdate()
plt.title('MA model predictions')
plt.show()

#Computing root mean squared error
    #Since pedictions is in form of series, RMSE cannot be applied in direct form
print("\n Root mean squared error for MA model\n")
#Dividing sqrt of dot product by no of observations
MA_rms = np.sqrt(np.dot(MA_predict, valid['Count']))/valid.shape[0]
print(MA_rms)





#Combining AR and MA model
model = ARIMA(Train_log, order=(1, 1, 1))  
results_ARIMA = model.fit()  
plt.plot(train_log_diff.dropna(),  label='original') 
plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 
plt.legend(loc='best')
plt.title('ARIMA model')
plt.show()

#Predicting Validation data using ARIMA model
ARIMA_predict=results_ARIMA.predict(start="2014-06-25", end="2014-09-25")
print("\n\nPredicting Validation data using ARIMA model")
print(ARIMA_predict.head(5))

#Finding cumulative sum of array elements at each index and assigning that as value for succeeding index
ARIMA_predict=ARIMA_predict.cumsum().shift().fillna(0)
print("\nAfter assiging cumulative sum of preceeding index value as the value for each index")
print(ARIMA_predict)
#Since first validation data is "ZERO", we add first valid data to all entries to compensate
    #Creating a series of first valid data filled equal to no of rows times of valid dataset
ARIMA_predict1=pd.Series( (np.ones(valid.shape[0]) * np.log(valid['Count'])[0]) , index = valid.index)
#Adding 2 series
ARIMA_predict1=ARIMA_predict1.add(ARIMA_predict,fill_value=0)
#Calculating expoenential- to compensate log of each value
ARIMA_predict = np.exp(ARIMA_predict1)

#Plottig ARIMA graph
fig,ax = plt.subplots()
plt.plot(valid['Count'], label = "Valid") 
plt.plot(ARIMA_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best')
#To avoid overlapping of xticks
fig.autofmt_xdate()
plt.title('ARIMA model predictions')
plt.show()

#Computing root mean squared error
    #Since pedictions is in form of series, RMSE cannot be applied in direct form
print("\n Root mean squared error for ARIMA model\n")
#Dividing sqrt of dot product by no of observations
ARIMA_rms = np.sqrt(np.dot(ARIMA_predict, valid['Count']))/valid.shape[0]
print(ARIMA_rms)

############################################################################################################################################################


                                                                #6.SARIMA model on daily time series
    #Extension of ARIMA; This takes seasonality also into account
fit1 = sm.tsa.statespace.SARIMAX(Train.Count, order=(1, 1, 1),seasonal_order=(1,1,1,7)).fit()
#To predict based on values out of trained model
y_hat['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True)

#Plotting graph
plt.figure(figsize=(16,8)) 
plt.plot( Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat['SARIMA'], label='SARIMAX') 
plt.legend(loc='best')
plt.title('SARIMA')
plt.show()


#Computing root mean squared error
print("\n Root mean squared error for SARIMA model on daily time series model\n")
SARIMA_rms = sqrt(mean_squared_error(valid.Count, y_hat.SARIMA)) 
print(SARIMA_rms)

############################################################################################################################################################

#Summarizing RMSE of all models
print("\n\n\nSummarizing RMSE of all models")
print('\nNaive Approach:\t ', format(Naive_rms))

print('\nMOVING AVERAGE')
print('\tLast 10 observations:\t', format(Mv_avg_10_rms))
print('\tLast 30 observations:\t', format(Mv_avg_30_rms))
print('\tLast 50 observations:\t', format(Mv_avg_50_rms))

print('\nSimple Exponential Smoothing:\t' ,format(SES_rms))

print('Holt’s Linear Trend Model:\t' ,format(Holt_Linear_rms))

print('Holt winter’s model:\t\t' ,format(Holt_Winter_rms))

print('\nARIMA model')
print('AR Model:\t' ,format(AR_rms))
print('MA Model:\t' ,format(MA_rms))
print('ARIMA Model:\t' ,format(ARIMA_rms))

print('SARIMAX model:\t' ,format(SARIMA_rms))

############################################################################################################################################################
