# Google Stock Price Predictor using LSTM

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict Google stock prices based on historical opening prices. The model uses time series analysis and deep learning techniques to forecast future stock prices.

## Features
- **Data Preprocessing**: Handles CSV data with date parsing and numeric conversion
- **Feature Scaling**: Applies MinMaxScaler for normalizing stock prices
- **LSTM Architecture**: 4-layer LSTM network with dropout regularization
- **Visualization**: Plots actual vs. predicted stock prices for comparison

## Requirements

### Dependencies
```
numpy
matplotlib
pandas
scikit-learn
keras
tensorflow
```

### Installation
```bash
pip install numpy matplotlib pandas scikit-learn keras tensorflow
```

## Project Structure
```
├── Stock_Market.ipynb          # Main Jupyter notebook with prediction model
├── Google_Stock_Price_Test.csv # Test dataset
├── README.md                    # Project documentation
```

## Data Format
The input CSV files should contain the following columns:
- **Date**: Trading date (used as index)
- **Open**: Opening price
- **High**: Daily high price
- **Low**: Daily low price
- **Close**: Closing price
- **Volume**: Trading volume

**Note**: Volume values contain commas (e.g., "1,657,300") which are handled in preprocessing.

## Model Architecture
The LSTM model consists of:
- 4 LSTM layers with 50 units each
- Dropout layers (0.2) after each LSTM layer for regularization
- Dense output layer with 1 unit
- Adam optimizer with mean squared error loss function

### Training Parameters
- **Epochs**: 100
- **Batch Size**: 32
- **Sequence Length**: 60 days
- **Features**: Single feature (opening price)

## Usage

### 1. Load and Prepare Data
```python
# Load training data
dataset = pd.read_csv('Google_Stock_Price_Train.csv', index_col="Date", parse_dates=True)

# Convert numeric columns
dataset["Close"] = dataset["Close"].str.replace(',', '').astype(float)
dataset["Volume"] = dataset["Volume"].str.replace(',', '').astype(float)
```

### 2. Scale Features
```python
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

### 3. Create Training Sequences
```python
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
```

### 4. Train Model
```python
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
```

### 5. Make Predictions
```python
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

### 6. Visualize Results
```python
plt.plot(real_stock_price, color='red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.legend()
plt.show()
```

## Results
The model generates predictions for stock prices and visualizes them against actual prices. The visualization helps evaluate the model's accuracy in capturing stock price trends.

## Limitations
- Uses only opening price for predictions (ignores high, low, close, volume)
- Requires sufficient historical data for meaningful predictions
- Stock market prediction is inherently uncertain and should not be used for financial decisions

## Future Improvements
- Incorporate multiple features (high, low, close, volume)
- Implement cross-validation
- Experiment with hyperparameter tuning
- Add performance metrics (RMSE, MAE)
- Include sentiment analysis from news/social media

## Acknowledgments
This project demonstrates the application of LSTM networks for time series forecasting in financial markets.
