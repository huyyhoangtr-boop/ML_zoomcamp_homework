import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

# Prepare the data
df0 = pd.read_csv("car_fuel_efficiency_2026.csv")


df0_object = df0[df0.columns[df0.dtypes == "object"]]

for i in list(df0_object.columns):
    df0[i] = df0[i].str.replace(" ", "_").str.lower()
    
df0 = df0[[ "engine_displacement", "horsepower", "vehicle_weight", "model_year", "fuel_efficiency_mpg"]]
print (df0)
sns.histplot((df0["fuel_efficiency_mpg"]), bins = 90)
plt.savefig("fuel_efficiency_mpg visualization.png")
plt.close()

# Question 1
columns_with_missing_values = df0.columns[df0.isnull().sum()>0]
print ("Question 1:", np.array(columns_with_missing_values))

#Question 2
med_horsepower = df0["horsepower"].median()
print ("Question 2 - med horsepower:", med_horsepower)


#Question 3
df = df0.copy()
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_train_mean = df_train.copy()
df_train_zero = df_train.copy()
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]


mean_train = df_train["horsepower"].mean()
df_train_mean["horsepower"] = df_train_mean["horsepower"].fillna(mean_train)
df_val_mean = df_val.copy()
df_val_zero = df_val.copy()
df_val_mean["horsepower"] = df_val_mean["horsepower"].fillna(mean_train)


target_train = (df_train["fuel_efficiency_mpg"].values)
target_val = (df_val["fuel_efficiency_mpg"].values)
target_test = (df_test["fuel_efficiency_mpg"].values)

del df_train_zero["fuel_efficiency_mpg"]
del df_train_mean["fuel_efficiency_mpg"]
del df_val_zero["fuel_efficiency_mpg"]
del df_val_mean["fuel_efficiency_mpg"]

#print (df_train_mean)

df_train_mean = np.column_stack([np.ones(df_train_mean.shape[0]), df_train_mean])
df_val_mean = np.column_stack([np.ones(df_val_mean.shape[0]), df_val_mean])


W_mean = np.dot(np.dot(np.linalg.inv(np.dot(df_train_mean.T,df_train_mean)),df_train_mean.T),target_train)
#print(W_mean)
Y_mean = np.dot(df_val_mean, W_mean)
RMSE = root_mean_squared_error(Y_mean, target_val)
print("RMSE if mean is replaced", RMSE.__round__(3))



df_val_zero["horsepower"] = df_val_zero["horsepower"].fillna(0)
df_val_zero = np.column_stack([np.ones(df_val_zero.shape[0]), df_val_zero])

df_train_zero["horsepower"] = df_train["horsepower"].fillna(0)
df_train_zero = np.column_stack([np.ones(df_train_zero.shape[0]), df_train_zero])
W_zero = np.dot(np.dot(np.linalg.inv(np.dot(df_train_zero.T,df_train_zero)),df_train_zero.T),target_train)
Y_zero = np.dot(df_val_zero, W_zero)
rmse = root_mean_squared_error(Y_zero,target_val)

print("RMSE if 0 is replaced", rmse.__round__(3))
if rmse < RMSE:
    print ("Question 3 - better rmse is when fillna(0)")
elif rmse > RMSE: 
    print ("Question 3 - better rmse is when fillna(mean)")
else:
    print ("Question 3 - both are equally good")


# Question 4
XTX = np.dot(df_train_zero.T,df_train_zero)
r = [0, 0.01, 0.1,1, 5, 10, 100]
category_rmse = {}
for i in r:
    XTX1 = XTX +  (np.eye (XTX.shape[0]))*i
    W = np.dot(np.dot(np.linalg.inv(XTX1),df_train_zero.T),target_train)
    Y = np.dot(df_val_zero, W)
    rmse = root_mean_squared_error(Y, target_val )
    category_rmse[i] = (rmse.__round__(4))
#print (category_rmse)

for r, rmse in category_rmse.items():
    if category_rmse[r] ==  min(category_rmse.values()):
        print ("Question 4 - best r:", r)
        break
    


#Question 5
df_seed = df0.copy()
seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_list = []
for i in seed:
    np.random.seed(i)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df_seed.iloc[idx[:n_train]]
    df_val = df_seed.iloc[idx[n_train:n_train + n_val]]
    df_test = df_seed.iloc[idx[n_train + n_val:]]


    target_train = (df_train["fuel_efficiency_mpg"].values)
    target_val = (df_val["fuel_efficiency_mpg"].values)
    target_test = (df_test["fuel_efficiency_mpg"].values)

    del df_train["fuel_efficiency_mpg"]
    del df_val["fuel_efficiency_mpg"]
    del df_test["fuel_efficiency_mpg"]
    
    df_train = df_train.fillna(0)
    df_val = df_val.fillna(0)
    df_test = df_test.fillna(0)
    #Question 6
    if i == 9:
            df_test = np.column_stack([np.ones(df_test.shape[0]), df_test])
            df_train_val = np.concatenate([df_train, df_val])
            df_train_val = np.column_stack([np.ones(df_train_val.shape[0]), df_train_val])
            
            target_train_val = np.concatenate([target_train, target_val])
            W_9 = np.dot(np.dot(np.linalg.inv(np.dot(df_train_val.T,df_train_val) + 0.001 * np.eye(df_train_val.shape[1])),df_train_val.T),target_train_val)
            Y_9 = np.dot(df_test, W_9)
            rmse_test = root_mean_squared_error(Y_9, target_test)
            print ("Question 6 - RMSE on test set", rmse_test)

    df_train = np.column_stack([np.ones(df_train.shape[0]), df_train])
    df_val = np.column_stack([np.ones(df_val.shape[0]), df_val])
    

    W = np.dot(np.dot(np.linalg.inv(np.dot(df_train.T,df_train)),df_train.T),target_train)
    Y = np.dot(df_val, W)
    rmse = root_mean_squared_error(Y, target_val )
    #print ("Seed", i, "RMSE", rmse.__round__(4))
    rmse_list.append(rmse)
    
    
    
    
    
    
rmse_list = np.array(rmse_list)
print ("Question 5 - std of RMSE", np.std(rmse_list).round(3))    

