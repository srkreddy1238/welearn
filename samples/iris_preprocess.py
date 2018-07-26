""" Preprocessing API sample """

# Standard imports
import pandas as pd
import numpy as np

#Import utils
import sys
sys.path.insert(0,'../util')
import util


# Load data frame
df = pd.read_csv("data/iris.csv")

# Have a look
print(df.head())
print(df.info())

# Dependent Variable is : Species

# IQR Processing
df = util.drop_by_iqr(df, ['Species'])
print(df.shape)

# Scaling and split
X_train, Y_train, X_test, Y_test = util.split_dataframe(df, 'Species', scale=True, factor=0.8, seed=10)
