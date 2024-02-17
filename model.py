import yfinance as yf
import numpy as np
from fastdtw import fastdtw
from sklearn.model_selection import KFold, GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier


# Computes the Dynamic Time Warping (DTW) distance between two stock time series.
# The lower the result, the more similar the two stock patterns are.
def dtw_distance(stock_1, stock_2):
 distance, _ = fastdtw(stock_1, stock_2)
 return float(distance)


# To download Stock Data
# We'll use yfinance to get the closing prices of various companies for a specified time period.
tickers = ["AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "JPM", "JNJ", "V", "PG", "NVDA", "HD", "UNH", "PYPL"]
start_date = "2022-01-01"
end_date = "2023-01-01"
data = {ticker: yf.download(ticker, start=start_date, end=end_date)['Close'].tolist() for ticker in tickers}


# Prepare data for our model
X = [data[ticker] for ticker in tickers]
y = tickers


# Calculate Similarities Between Stocks
# We're going to compare every stock's price patterns with every other stock's patterns.
dtw_matrix = np.zeros((len(tickers), len(tickers)))
for i in range(len(tickers)):
    for j in range(len(tickers)):
        dtw_matrix[i, j] = dtw_distance(X[i], X[j])

print("\nHow Similar Are These Stocks?")
print(dtw_matrix)


# Find Most Similar Stocks
# Using K-nearest neighbours with our DTW distance metric to find the most similar stocks.
knn = KNeighborsClassifier(metric=dtw_distance)


# Using KFold to split our data into parts (or "folds") for cross-validation.
cv = KFold(n_splits=3, shuffle=True, random_state=42)


# Let's try different values for K (number of neighbors) to find the best one.
param_grid = {'n_neighbors': list(range(1, min(len(tickers)-1, 10)))}


# GridSearchCV will try out each value of K and tell us the best one.
grid_search = GridSearchCV(knn, param_grid, cv=cv)
grid_search.fit(X, y)

print(f"\nBest Number of Neighbors: {grid_search.best_params_['n_neighbors']}")


# For each stock, let's find out which other stock is most similar to it.
distances, neighbors_idx = grid_search.best_estimator_.kneighbors(X, 2)
for i, ticker in enumerate(tickers):
 print(f"\n{ticker}'s closest buddy is {tickers[neighbors_idx[i][1]]} with a similarity score of {distances[i][1]}.")