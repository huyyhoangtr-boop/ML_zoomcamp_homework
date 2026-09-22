from ast import If

import numpy as np
import pandas as pd


# Question 1: pandas and numpy versions
pd.__version__
print("Pandas version:", pd.__version__)

#Question 2 (how many records)

df = pd.read_csv("car_fuel_efficiency_2026.csv")
print("Number of records in the dataset:", df.shape[0])

# Question 3 (how many fuel types )
print("Number of unique fuel types:", df["fuel_type"].nunique())

# Question 4 (how many rows with null value)
print("Number of rows with null values:", (df.isnull().sum() > 0).sum())

# Question 5 (The maximum fuel efficiency in Asia )
print("Maximum fuel efficiency in Asia:", df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max())

# Question 6 (The change of med value) 
Initial_med = df["horsepower"].median()
Mode = df["horsepower"].mode()
df["revised_horsepower"] = df["horsepower"].fillna(Mode[0])

Final_med = df["revised_horsepower"].median()
print("Initial median value of horsepower:", Initial_med)
print("Final median value of horsepower after filling missing values:", Final_med)
if Initial_med > Final_med:
    print("-->The med value decreased")
elif Initial_med < Final_med:
    print("-->The med value increased")
else:
    print("-->The med value didn't change.")
    




# Question 7

a = df[
    (df["origin"] == "Asia")
]
a = a[["vehicle_weight", "model_year"]]



X = a.head(7).values
XT = X.T
XTX = np.dot(XT, X)
inv_XTX = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
k = np.dot(inv_XTX, XT)
w = np.dot(k, y)
print(" The sum of all the elements of w:", np.sum(w))

