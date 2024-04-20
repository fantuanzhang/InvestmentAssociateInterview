import numpy as np
from numpy import genfromtxt
from scipy.stats import rankdata, norm
import os
import csv
import pandas as pd
cwd = os.getcwd()

# question 1
np.random.seed(1)

X = np.random.normal(0, 1,5000)
Y_temp = np.random.normal(0, 1,5000)

# Set target correlation
correlation_target = 0.5

# Generate adjusted Y
Y = X * correlation_target + Y_temp * np.sqrt(1 - correlation_target**2)

# Verify the correlation
correlation = np.corrcoef(X, Y)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

# question 2, I find some of the data can not be download from yahoo finance, so I download the data from MarketWatch.
# since the historical data download is limited one year on MarketWatch for me. I used historical data from 2022-12-29 to 2023-12-29

USDCADFX_file = cwd + "//input_data//CADUSDFX.csv"
SP500_file = cwd + "//input_data//SP500.csv"

USDCADFX_data = []
SP500_data = []
USDCADFX = []
SP500 = []
with open(USDCADFX_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        USDCADFX_data.append(row)

with open(SP500_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        SP500_data.append(row)

# clean the data, set USDCADFX as USD/CAD FX daily closed data, SP500 as  S&P500 daily closed data
for i in range(1, len(USDCADFX_data)):
    USDCADFX.append(float(USDCADFX_data[i][4]))

for i in range(1, len(SP500_data)):
    SP500.append(float(SP500_data[i][4]))

# transform the list data type to numpy array data type
USDCADFX_original = np.array(USDCADFX)
SP500_original = np.array(SP500)

# Compute the original empirical data set covariance matrix
original_cov = np.cov(USDCADFX_original, SP500_original)

# Perform Cholesky decomposition on the original covariance matrix
L = np.linalg.cholesky(original_cov)

# Define the target correlation and covariance matrix
target_correlation = 0.5
var_x = original_cov[0, 0]
var_y = original_cov[1, 1]
target_cov = np.array([
    [var_x, np.sqrt(var_x * var_y) * target_correlation],
    [np.sqrt(var_x * var_y) * target_correlation, var_y]
])

# Cholesky decomposition of the target covariance matrix
L_target = np.linalg.cholesky(target_cov)


# Transform the data
data = np.vstack((USDCADFX_original, SP500_original))
transformed_data = L_target @ np.linalg.inv(L) @ data
USDCADFX_modified = transformed_data[0]
SP500_modified = transformed_data[1]

# Verify the new correlation
new_correlation_matrix = np.corrcoef(transformed_data)



#----------------------------------------------------------------------------------------------------------------------------------------------------------------
# question 3, extend the above analysis to 4 dimension. I downloaded the NASDAQ inedx and Gold future price as input data

NASDAQ_file = cwd + "//input_data//NASDAQ.csv"
Gold_file = cwd + "//input_data//Gold.csv"

NASDAQ_data = []
Gold_data = []
NASDAQ = []
Gold = []
with open(NASDAQ_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        NASDAQ_data.append(row)

with open(Gold_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        Gold_data.append(row)

# clean the data, set NASDAQ as NASDAQ daily closed data, Gold as  Gold daily closed data
for i in range(1, len(NASDAQ_data)):
    NASDAQ.append(float(NASDAQ_data[i][4]))

for i in range(1, len(Gold_data)):
    Gold.append(float(Gold_data[i][4]))

# transform the list data type to numpy array data type
NASDAQ_original = np.array(NASDAQ)
Gold_original = np.array(Gold)

# now we have USDCADFX_original, SP500_original, NASDAQ_original and Gold_original

# Stack the data into a matrix
data_4dim = np.vstack((USDCADFX_original, SP500_original, NASDAQ_original, Gold_original))

# Compute the empirical covariance matrix
original_cov_4dim = np.cov(data_4dim)

# Perform Cholesky decomposition on the original covariance matrix
L_original_4dim = np.linalg.cholesky(original_cov_4dim)

# Define the target correlations
target_correlations_4dim = np.array([
    [1.0, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5],
    [0.5, 0.5, 0.5, 1.0]
])

# Construct the target covariance matrix using original variances
variances = np.diag(original_cov_4dim)
target_cov_4dim = target_correlations_4dim * np.sqrt(np.outer(variances, variances))

# Cholesky decomposition of the target covariance matrix
L_target_4dim = np.linalg.cholesky(target_cov_4dim)

# Transform the data
transformed_data_4dim = L_target_4dim @ np.linalg.inv(L_original_4dim) @ data_4dim

# Verify the new covariance matrix
new_covariance_matrix_4dim = np.cov(transformed_data_4dim)

USDCADFX_modified = transformed_data_4dim[0]
SP500_modified = transformed_data_4dim[1]
NASDAQ_modified = transformed_data_4dim[2]
Gold_modified = transformed_data_4dim[3]

print("ok")


