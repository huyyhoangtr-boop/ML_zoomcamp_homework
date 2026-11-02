import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv("car_fuel_efficiency_2026.csv")


#FILL MISSING:
for i in df.columns:
    if df[i].isnull().sum() >0: df[i] = df[i].fillna(0)
    
    
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)
target_full_train = df_full_train["fuel_efficiency_mpg"]
target_train = df_train["fuel_efficiency_mpg"]
target_val = df_val["fuel_efficiency_mpg"]
target_test = df_test["fuel_efficiency_mpg"]


del df_full_train["fuel_efficiency_mpg"]
del df_train["fuel_efficiency_mpg"]
del df_test["fuel_efficiency_mpg"]
del df_val["fuel_efficiency_mpg"]
    


# dv
from sklearn.feature_extraction import DictVectorizer 
dv = DictVectorizer(sparse=True)
df = df.to_dict(orient = "records")
dv.fit(df)
def dv_turn(df):
    df = df.to_dict(orient = "records")
    x = dv.transform(df)
    return x
x_train = dv_turn(df_train)
x_val = dv_turn(df_val)
x_test = dv_turn(df_test)
x_full_train = dv_turn(df_full_train)

#Q1:
from sklearn.tree import DecisionTreeRegressor, export_text
model_tree = DecisionTreeRegressor(max_depth=1)
model_tree.fit(x_train, target_train)
print(export_text(model_tree, feature_names= dv.get_feature_names_out()))
print()

#Q2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error as rmse
model_forest = RandomForestRegressor(n_estimators=10,random_state=1,n_jobs=-1)
model_forest. fit(x_train, target_train)
y_pred_val = model_forest.predict(x_val)
print(rmse(target_val, y_pred_val))
print(y_pred_val)
print()
#Q3
dict_rmse = {}
for i in [10, 50, 100, 150]:
    model_forest = RandomForestRegressor(n_estimators=i,random_state=1,n_jobs=-1)
    model_forest. fit(x_train, target_train)
    y_pred = model_forest.predict(x_val)
    dict_rmse["%s_%s" % ('n_estimators', i)] = rmse(target_val, y_pred)

print (min(dict_rmse, key = dict_rmse.get))
print()

#Q4
list_rmse = []
dict_rmse_q4 = {}
for depth in  [10, 15, 20, 25]:
    for estimators in [10, 50, 100, 150]:
        model_forest = RandomForestRegressor(n_estimators=estimators, max_depth=depth,random_state=1,n_jobs=-1)
        model_forest. fit(x_train, target_train)
        y_pred_q4 = model_forest.predict(x_val)
        list_rmse.append(rmse(target_val,y_pred_q4))
    dict_rmse_q4 ["%s_%s" % ("max_depth", depth)]= (np.array(list_rmse)).mean()
    
print (dict_rmse_q4)
print (min(dict_rmse_q4, key = dict_rmse_q4.get))

#Q5
model_forest = RandomForestRegressor(n_estimators=10,random_state=1,n_jobs=-1, max_depth=20)
model_forest. fit(x_train, target_train)
feature = dv.get_feature_names_out()
num = 0
dictionary = {}
for i in model_forest.feature_importances_:
    dictionary[feature[num]] = i
    num+=1
for items, key in dictionary.items():
    for i in [ "engine_displacement", "acceleration", "horsepower", "vehicle_weight"]:
        if items == i: print (dictionary[items])
        
import xgboost as xgb
for i in [0.3,0.1]:
    X_train_xgb = xgb.DMatrix(x_train, target_train,feature_names= list(feature))
    X_val_xgb = xgb.DMatrix(x_val, target_val,feature_names= list(feature))# enable_categorical = True)
    xgb_params = {
            "eta" :i, #How fast the model learn
            "max_depth" :6,
            "min_child_weight":1,
            "objective": "reg:squarederror",
            "nthread":8,
            "seed": 1,
            "verbosity":1
        }
    model = xgb.train(xgb_params, X_train_xgb, num_boost_round = 100, verbose_eval =1,evals = [(X_train_xgb, "train"), (X_val_xgb, "val")])