# Final Project: Analysis of Oil, Gold, and Silver Prices

***Table of Contents***

Project Overview

Problem Statements

Datasets and Sources

Data Preparation

Exploratory Data Analysis (EDA)

Statistical Analysis and Hypothesis Testing

Predictive Modeling

Results

Conclusion

How to Run the Project

Presentation

Acknowledgements



***Project Overview***

The goal of this project is to perform an end-to-end data analysis on price data of three key commodities: Oil, Gold, and Silver. 
The project applies various data analysis techniques, including exploratory data analysis (EDA), statistical hypothesis testing, 
and predictive modeling using different machine learning algorithms. 
The results provide insights into the relationships between these commodities, their market behavior over time, and how their prices can be predicted using various models.

***Problem Statements***

***1. Influence of Commodity Prices on Markets:***
How do the prices of commodities like oil, gold, and silver impact financial markets, and how can this knowledge guide better investment decisions?


***2. Impact of Price Volatility:***
How does the volatility in oil prices affect the prices of other commodities like gold and silver?


***3. Predicting Future Price Movements:***
Can the price movements of oil and gold serve as early indicators for future changes in silver prices?


***Datasets and Sources***

Daily Silver Prices : https://www.kaggle.com/datasets/balabaskar/silver-prices-1968-2021

Gold Prices: https://www.kaggle.com/code/daniilkrasnoproshin/random-forest-for-regression-problems/input

Brent Oil Prices: [Kaggle](https://www.kaggle.com/datasets/mabusalah/brent-oil-prices)

***Data Preparation***


The data preparation phase involved several key steps:

***Loading the datasets:*** The datasets for oil, gold, and silver prices were loaded from CSV files.
***Data Cleaning:*** Dates were converted to a standard datetime format, and irrelevant columns were removed.
***Data Merging:*** The datasets were merged into a single DataFrame based on the 'Date' column.
***Data Transformation:*** Price values were rounded to three decimal places for consistency.

# Example code for data loading and preparation
brent_oil_data = pd.read_csv('Data/BrentOilPrices.csv')
gold_data = pd.read_csv('Data/gld_price_data.csv')
silver_data = pd.read_csv('Data/LBMA-SILVER.csv')

# Convert dates and clean data
brent_oil_data['Date'] = pd.to_datetime(brent_oil_data['Date'], errors='coerce')
gold_data['Date'] = pd.to_datetime(gold_data['Date'], format='%m/%d/%Y')
silver_data['Date'] = pd.to_datetime(silver_data['Date'])

# Merge datasets
merged_data = pd.merge(brent_oil_data, gold_data, on='Date', how='inner')
merged_data = pd.merge(merged_data, silver_data, on='Date', how='inner')

# Save cleaned data
merged_data.to_csv('Clean Data/merged_data.csv', index=False)

***Exploratory Data Analysis (EDA)***
During the EDA phase, we explored the following:

***Distribution of Commodity Prices:*** Visualized the distribution of oil, gold, and silver prices.

***Price Trends Over Time:*** Analyzed the trend of prices from 2008 to 2018.

***Correlation Analysis:*** Examined the correlation between the prices of oil, gold, and silver.

***Key Findings:***

***Oil prices:*** Showed significant volatility with two peaks, indicating market instability during certain periods.

***Gold prices:*** Were relatively stable with a peak around 2011-2012, possibly due to economic uncertainty.

***Silver prices:*** Displayed a sharp peak around 2011, similar to gold, but with greater overall volatility.

## Statistical Analysis and Hypothesis Testing

***We tested three main hypotheses using statistical methods:***

***Gold as a Hedge:*** Tested the correlation between gold prices and oil price volatility. 

***Result:*** A negative correlation, indicating that gold does not consistently act as a hedge against oil market volatility.

***Impact of Oil Price on Other Commodities:*** Tested the linear relationship between oil and gold/silver prices. Result: A significant positive correlation, especially between oil and silver.

***Predicting Future Prices:*** Analyzed time-lagged correlations to see if oil and gold prices could predict silver prices. 
***Result:*** Strong positive correlations, supporting the hypothesis.


# Example code for hypothesis testing
from scipy.stats import pearsonr
corr_gold_oil, p_value_gold_oil = pearsonr(merged_data['GLD'], merged_data['OIL'])

***Predictive Modeling***
We implemented three models for forecasting future prices:

Prophet: Used for trend-based forecasting, capturing long-term trends but with higher uncertainty.
ARIMA: Provided the best performance for silver price prediction with the lowest RMSE.
LSTM: Captured short-term fluctuations effectively.
Model Performance:
Prophet Model (Oil): RMSE = 8.71, MAE = 6.38
ARIMA Model (Silver): RMSE = 0.61, MAE = 0.32
LSTM Model (Silver): RMSE = 0.78, MAE = 0.65


Results


The analysis revealed key insights into how the prices of oil, gold, and silver are interrelated. The models demonstrated varying degrees of success in forecasting future prices, with ARIMA being the most accurate for silver price prediction.

***Conclusion***
This project demonstrated the value of using multiple analytical approaches to understand and predict commodity prices. The strong relationships identified between these commodities can guide better investment decisions and highlight the importance of model selection in financial forecasting.

***How to Run the Project***
Clone the repository: git clone [repository link]
Install the required packages: pip install -r requirements.txt
Run the main script: python main_script.py
View the presentation: Presentation Link


***Presentation's link:*** https://www.canva.com/design/DAGOoyFTkWg/zYYAvpmCNK2fXB42uqvQAQ/edit?utm_content=DAGOoyFTkWg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
