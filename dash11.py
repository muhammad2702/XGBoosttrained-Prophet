# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:37:39 2024

@author: atom
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 01:22:19 2024

@author: atom
"""
from PIL import Image, ImageDraw, ImageFont

import base64
import io
from twelvedata import TDClient
import pandas as pd
import numpy as np
np.float_ = np.float64

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import seaborn as sns
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

class DataFeeder:
    def __init__(self, ticker, api_key, timeframe):
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = timeframe
        self.td = TDClient(apikey=self.api_key)
        try:
            self.df = self.fetch_ticker()
            print(f"Data fetched for {self.ticker}")
        except Exception as e:
            print("nope", e)

    def fetch_ticker(self):
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=210,
            timezone="America/New_York",
        )
        df = ts.as_pandas()
        #print(df)
        df = df.reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        return df

class Indicators:
    def __init__(self, data_feeder):
        self.df = data_feeder

    def calculate_sma(self, periods):
        for period in periods:
            self.df[f'SMA_{period}'] = self.df['close'].astype(float).rolling(window=period).mean()
            self.df[f'SMA_{period}'].fillna(method='bfill', inplace=True)

    
    def calculate_stochastic(self, periodK=12, smoothK=3, periodD=1, periodD1=4):
        # Calculate lowest low and highest high over periodK
        self.df['lowest_low'] = self.df['low'].rolling(window=periodK).min()
        self.df['highest_high'] = self.df['high'].rolling(window=periodK).max()
        
        # Calculate %K
        self.df['%K'] = 100 * (self.df['close'] - self.df['lowest_low']) / (self.df['highest_high'] - self.df['lowest_low'])
        
        # Smooth %K with an SMA over smoothK
        self.df['%K_smooth'] = self.df['%K'].rolling(window=smoothK).mean()
        
        # Calculate %D as the SMA of %K_smooth over periodD
        self.df['%D'] = self.df['%K_smooth'].rolling(window=periodD).mean()
        
        # Calculate %D1 as the SMA of %K_smooth over periodD1
        self.df['%D1'] = self.df['%K_smooth'].rolling(window=periodD1).mean()
        
        # Calculate Double Slow K Denominator (highest - lowest over periodK)
        self.df['DoubleSlowKDen'] = self.df['%K_smooth'].rolling(window=periodK).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Calculate Double Slow D Denominator (highest - lowest over periodK)
        self.df['DoubleSlowDDen'] = self.df['%D'].rolling(window=periodK).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Calculate Double Slow K
        self.df['DoubleSlowK'] = 100 * (self.df['%K_smooth'] - self.df['%K_smooth'].rolling(window=periodK).min()) / self.df['DoubleSlowKDen']
        
        # Calculate Double Slow D
        self.df['%FastD'] = self.df['%D'] 
        
        self.df['DoubleSlowD'] = 100 * (self.df['%D'] - self.df['%D'].rolling(window=periodK).min()) / self.df['DoubleSlowDDen']
        self.df.fillna(method='bfill', inplace=True) 
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)

class Predictor:
    def __init__(self, data_feeder, freq, timeframe):
        self.freq = freq
        self.df = data_feeder
        self.future_datapoint = 5
        self.timeframe = timeframe
  
    def load_model(self, model_path):
        # Load the pretrained XGBoost model from JSON
        model = xgb.XGBRegressor()
        model.load_model(model_path)  # Load model from JSON
        return model

    def predict_with_xgb(self, model, features):
        # Convert features to DMatrix
        #dmatrix = xgb.DMatrix(features)
        # Use the pretrained model to predict
        predictions = model.predict(features)
        return predictions

    def run_prediction(self, data_feeder, model_paths):
        df_predictsx = pd.DataFrame()
        feature_columns = [col for col in data_feeder.columns if col in ['predicted_open', 'predicted_high', 'predicted_low', 'predicted_close', 'predicted_SMA_21', 'predicted_SMA_50',
               'predicted_SMA_200', 'predicted_SMA_36', 
               'predicted_%FastD', 'predicted_DoubleSlowK',]]
        print("feature_columns")
        print(feature_columns)
        # Prepare features from the dataframe
        features = data_feeder[feature_columns].fillna(0)  # Fill NaNs or preprocess as needed
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        for model_path in model_paths:
            model = self.load_model(model_path)  # Load each model
            predictions = self.predict_with_xgb(model, features)  # Get predictions
            predictions = [self.s_to_time(sec) for sec in predictions]

            df_predictsx[f'predictions_{model_path}'] = predictions  # Store predictions with model name as column
        print("df_predictsx")
        print(df_predictsx)
        return df_predictsx
    
    def s_to_time(self , seconds):
     # Calculate hours, minutes, and remaining seconds
     hours = int(seconds) // 3600
     minutes = (int(seconds) % 3600) // 60
     seconds = int(seconds) % 60
     
     # Create a time object
     return pd.to_datetime(f'{hours:02}:{minutes:02}:{seconds:02}', format='%H:%M:%S').time()


    def check_stationarity(self, timeseries):
          """
          Perform Augmented Dickey-Fuller test to check stationarity
          """
          print('Results of Dickey-Fuller Test:')
          dftest = adfuller(timeseries, autolag='AIC')
          dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
          for key,value in dftest[4].items():
              dfoutput['Critical Value (%s)'%key] = value
          print(dfoutput)
          
          if dftest[1] <= 0.05:
              print("Time series is stationary")
          else:
              print("Time series is not stationary")
    
    def make_stationary(self, timeseries):
        """
        Difference the series until it becomes stationary
        """
        diff = timeseries
        for i in range(10):  # max 10 differences
            if adfuller(diff, autolag='AIC')[1] > 0.05:
                diff = diff.diff().dropna()
            else:
                print(f"Series became stationary after {i+1} differences")
                return diff
        print("Series did not become stationary after 10 differences")
        return diff
    
    def autocorrelation_analysis(self, timeseries):
        """
        Perform autocorrelation and partial autocorrelation analysis
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        sm.graphics.tsa.plot_acf(timeseries, lags=40, ax=ax1)
        ax1.set_title("Autocorrelation")
        
        sm.graphics.tsa.plot_pacf(timeseries, lags=40, ax=ax2)
        ax2.set_title("Partial Autocorrelation")
        
        plt.tight_layout()
        plt.show()
  
    def covariance_analysis(self, df):
        """
        Perform covariance analysis on the dataframe
        """
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    
    def prepare_data(self):
        """
        Prepare the data for analysis and prediction
        """
        # Check stationarity of close prices
        print("Checking stationarity of close prices:")
        self.check_stationarity(self.df['close'])
  
        # Make the series stationary if it's not
        stationary_close = self.make_stationary(self.df['close'])
        stationary_open = self.make_stationary(self.df['open'])
        stationary_low = self.make_stationary(self.df['low'])
        stationary_high = self.make_stationary(self.df['high'])
  
        # Perform autocorrelation analysis
        print("Performing autocorrelation analysis:")
        self.autocorrelation_analysis(stationary_close)
  
        # Perform covariance analysis
        print("Performing covariance analysis:")
        self.covariance_analysis(self.df)
  
        # Update the dataframe with the stationary series
        self.df['stationary_close'] = stationary_close
        self.df['stationary_open'] = stationary_open
        self.df['stationary_low'] = stationary_low
        self.df['stationary_high'] = stationary_high
  
        return self.df
          
    def time_to_s(self, time_series):
        return time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
    
    def future_dates(self, y):
      
        last_date = y.iloc[-1]


        # Determine the time delta in seconds based on timeframe
        if self.timeframe == '1min':
            delta_seconds = 60
                
        elif self.timeframe == '5min':
            delta_seconds = 300
        elif self.timeframe == '15min':
            delta_seconds = 900
        else:
            raise ValueError("Unsupported timeframe")
        
        future_dates_seconds = [last_date + delta_seconds * i for i in range(1, self.future_datapoint + 1)]
        

        future_dates_series = pd.Series(future_dates_seconds)
        extended_series = pd.concat([y, future_dates_series], ignore_index=True)

        return extended_series
    
        
    def prophet(self):
        df_predicts = pd.DataFrame()
        for feature in self.df.columns :
            if feature != 'date' and feature !='time_pd':
                prophet_df = pd.DataFrame({ 
                'ds' : self.df['date'] ,
                'y'  : self.df[feature]
                })
                
                self.df[feature].dropna()
                model = Prophet() 

                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=self.future_datapoint,  freq= self.freq) # i collected 40 from twevle api and 1 period means ill gwt 41 dates and the duration between them is #period  = datapoints
                # i need 5 data points into future so 5 additonoal point
               
                forecast = model.predict(future)
                #print(forecast)
                forecasted_values = forecast['yhat'].values
                df_predicts[f'predicted_{feature}'] = forecasted_values
                
        print("i am df_pre")
        print(df_predicts.columns)
        return df_predicts
    def detect_anomalies(self , df_predicts):
        #inputs to model
        
        input_to_model  = df_predicts
       
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_to_model)
        
        
        # y dates handling
    
        self.df['time_pd'] = pd.to_datetime(self.df['date'])
        seconds = self.time_to_s( self.df['time_pd'])
        y = self.future_dates(seconds)
        
        #append new dates according to 1 min timeframe and future datapoints
        #print("i am y")
        #print(y)
        combined_data = df_predicts
        
        # Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        print(scaled_data.shape)
        # Perform anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(scaled_data)
        
        # Add anomaly labels to the dataframe
        df_predicts['is_anomaly'] = anomalies
        anomalous_record = df_predicts[df_predicts['is_anomaly'] == -1]
        print("anomaly" )
        print(anomalous_record)
        #return df_predicts[df_predicts['is_anomaly'] == -1]
        return anomalous_record
        

    def visualize(self, df_predicts, column, anomalies):
    # Ensure the dataframe is sorted by date
        
        # Reset the index to create a numeric index
        df_predicts = df_predicts.reset_index(drop=True)
        
        plt.figure(figsize=(12, 6), dpi=150)
        
        # Plot the entire data series
        plt.plot(df_predicts.index[self.future_datapoint:], df_predicts[column][self.future_datapoint:], color='blue', label='Historical Data')
        plt.plot(df_predicts.index[-self.future_datapoint:], df_predicts[column][-self.future_datapoint:], color='limegreen', linewidth=3,   marker='o', markersize=8, label='Predicted Data')

        # Calculate the anomaly mask for the future datapoints
        future_data = df_predicts.iloc[self.future_datapoint:]
        anomaly_mask = future_data['is_anomaly'] == -1
        
        # Plot the anomalies using scatter
        if anomaly_mask.any():
            anomaly_indices = future_data.index[anomaly_mask]
            plt.scatter(anomaly_indices, df_predicts.loc[anomaly_indices, column], 
                        color='red', label='Anomalies')
        
        plt.title(f'Time Series for {column}')
        plt.xlabel('Time Steps')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        
        # 
        plt.tight_layout()
        st.pyplot(plt)
    def feature_importance_for_anomalies(self, df_predicts):
        # Prepare the data
        features = [col for col in df_predicts.columns if col not in ['ds', 'is_anomaly']]
        X = df_predicts[features]
        y = df_predicts['is_anomaly']
    
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
    
        # Get feature importances
        importances = clf.feature_importances_
        feature_imp = dict(zip(features, importances))
    
        # Sort features by importance
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
    
        # Visualize feature importances
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_features)), [imp for _, imp in sorted_features])
        plt.xticks(range(len(sorted_features)), [feat for feat, _ in sorted_features], rotation=90)
        plt.title('Feature Importance for Anomaly Detection')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        st.pyplot(plt)
    
        # Print feature importances
        print("Feature Importance for Anomaly Detection:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")
    
        return feature_imp

    def run_anomaly_detection(self, df_predicts):
        anomalies = self.detect_anomalies(df_predicts)
        print(f"Number of anomalies detected: {len(anomalies)}")
        
        # Visualize anomalies for a few key features
        for column in df_predicts.columns:
            if column in ['predicted_open', 'predicted_high', 'predicted_low', 'predicted_close', 'predicted_SMA_21', 'predicted_SMA_50',
                   'predicted_SMA_200', 'predicted_SMA_36', 
                   'predicted_%FastD', 'predicted_DoubleSlowK',] : 

                self.visualize(df_predicts, column, anomalies)
        # Show feature importance for anomaly detection
        self.feature_importance_for_anomalies(df_predicts)
        

def main():
    st.title("Financial Data Analysis App")

    # User inputs
    ticker = st.text_input("Ticker Symbol", "QQQ")
    api_key = st.text_input("API Key", "64e5d79c83ed4ff49b32db1b1a60627d")
    timeframe = st.selectbox("Timeframe", ['1min', '5min', '15min'])
    freq = st.selectbox("Frequency for Prophet", ['1T', '5T', '15T'])

    # Fetch data
    if st.button("Fetch Data"):
        data_feeder = DataFeeder(ticker, api_key, timeframe)
        data = data_feeder.fetch_ticker()
        st.subheader("Fetched Data")
        st.write(data)

        # Prepare data and calculate indicators
        predictor = Predictor(data, freq, timeframe)
        prepared_data = predictor.prepare_data()
        indicators = Indicators(prepared_data)
        indicators.calculate_sma([21, 50, 200, 36])
        indicators.calculate_stochastic()

        # Forecast using Prophet
        predictions = predictor.prophet()
        predictor.run_anomaly_detection(predictions)
        st.subheader("Price Predictions")
        st.write(predictions)

        # Run prediction with models
        model_paths = [
            'xgboost_model_1-Long T1_60.json',
            'xgboost_model_1-Short T1_40.json',
            'xgboost_model_Long T1_60.json',
            'xgboost_model_Short T1_40.json'
        ]
        predictionsx = predictor.run_prediction(predictions, model_paths)
        print("predictionsx")
        print(predictionsx)
        for column in predictionsx.columns:
            try:
                # Check if the column is numerical and represents time
                if pd.api.types.is_numeric_dtype(predictionsx[column]):
                    # Convert nanoseconds to timedelta, then to a readable time
                    predictionsx[column] = pd.to_timedelta(predictionsx[column], unit='ns').dt.total_seconds()
                    predictionsx[column] = (pd.Timestamp('2024-08-10') + pd.to_timedelta(predictionsx[column], unit='s')).dt.time
                else:
                    print(f"Skipping column {column} because it is not numeric.")
            except Exception as e:
                # If the conversion fails, print the error and skip that column
                print(f"Skipping column {column}: {e}")

    # Display the DataFrame in Streamlit
        st.subheader("Predictions from the 2 minute trades trained model")
      
        
        last_five = predictionsx.tail(5)

# Convert to string if necessary
        last_five = last_five.astype(str)
        
        # Prepare text to write on the image
        text_to_write = last_five.to_string(index=False)
        
        # Create an image with a white background
        img = Image.new('RGB', (1500, 300), color='white')
        
        # Initialize the drawing context
        d = ImageDraw.Draw(img)
        
        # Load a font
        font = ImageFont.load_default()
        
        # Define starting position for the text
        text_position = (10, 10)
        
        # Write the text on the image
        d.text(text_position, text_to_write, fill=(0, 0, 0), font=font)
        
        # Save the image
        img_path = 'predictions_last_five.png'
        img.save(img_path)
        
        # Display the image in Streamlit
        st.image(img_path)

if __name__ == "__main__":
    main()
