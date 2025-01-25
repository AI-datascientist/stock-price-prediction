# Stock Price Prediction using LSTM

## Overview
This repository contains a **Stock Price Prediction** model that forecasts the next-day closing price using **historical stock market data**. The model leverages **Long Short-Term Memory (LSTM)** networks and incorporates **technical indicators** such as RSI, MACD, and Bollinger Bands to enhance predictive performance.

## Project Structure
```
📁 stock-price-prediction  
│── 📜 README.md                 # Project documentation  
│── 📂 data                      # Folder for storing stock data  
│── 📜 stock_prediction.ipynb    # Jupyter Notebook with full implementation  
│── 📜 report.pdf                # Technical report explaining the process  
│── 📂 models                    # Directory to store trained models  
│── 📂 scripts                   # Python scripts for modular execution  
```

## 1. Data Collection
- The dataset is fetched from **Yahoo Finance** using the `yfinance` library.
- Features used: **Open, High, Low, Close, Volume**.
- Technical indicators such as **RSI, MACD, and Bollinger Bands** are included.

### Run Data Retrieval
```bash
python scripts/data_fetch.py
```

## 2. Feature Engineering
- Technical indicators are calculated to improve the model’s accuracy.
- Missing values are handled, and data is normalized using `MinMaxScaler`.

## 3. Model Development
- **LSTM (Long Short-Term Memory)** model is trained to predict stock prices.
- The dataset is split into **training (80%)** and **testing (20%)**.
- **Dropout layers** are added to prevent overfitting.

### Run Model Training
```bash
python scripts/train_model.py
```

## 4. Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures absolute prediction error.
- **MAPE (Mean Absolute Percentage Error)**: Evaluates prediction accuracy.
- **R² Score**: Determines how well the model explains variance in stock prices.

### Run Model Evaluation
```bash
python scripts/evaluate.py
```

## 5. Installation & Dependencies

### Clone the Repository
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebook (Optional)
```bash
jupyter notebook
```

## 6. Results & Insights
### Key Findings:
- **LSTM performed well** with an RMSE of **X.XX** and an R² score of **X.XX**.
- The model successfully identified price trends based on **technical indicators**.
- Potential improvements include **transformer models and hybrid approaches**.

### Predicted vs. Actual Closing Price Chart
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Prices", color='blue')
plt.plot(predicted_prices, label="Predicted Prices", color='red')
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.legend()
plt.show()
```

