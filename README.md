# Missing-data-imputation

Here we provide novel missing data imputation methods for any incomplete univariate time-series dataset.
First we decompose the time series in order to get the seasonality, trend and irregular components.
Then, we select two query: front-query and back-query.
It is significant to select the most appropriate size of the front and back query.
In order to measure the similarity, we select the MAE as our distance metric. 
