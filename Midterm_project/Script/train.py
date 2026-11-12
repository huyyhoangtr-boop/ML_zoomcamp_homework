import numpy as np
import pandas as pd


import kagglehub
print()
import os
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
dv= DictVectorizer(sparse= True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Download latest version
path = kagglehub.dataset_download("evgenyarbatov/sapa-weather")
# Check the downloaded file
files = [f for f in os.listdir(path) if f.endswith(".parquet")]
df = pd.read_parquet(os.path.join(path, files[0]))
df = df.drop(columns=["timestamp","lat", "lon", "site_id", "source", "schema_version", "weather_rain_1h", "weather_id", "weather_description", "timestamp_local"])
df['weather_main'] = df['weather_main'].str.lower()
k = list(np.unique(df['weather_main'].values))
dictionary = {}
for i in k:
  if i != "rain":
    dictionary[i] = 0
  else: dictionary["rain"] = 1
df['weather_main'] = df['weather_main'].map(dictionary)
df.isnull().sum()
for i in df.columns:
  if df[i].isnull().sum() >0: df[i]=df[i].fillna(df[i].mean())
  

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
target_train = df_train.weather_main.values
target_val = df_val.weather_main.values
target_test = df_test.weather_main.values
target_full_train = df_full_train.weather_main.values
df_full_train.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
del df_train['weather_main']
del df_val['weather_main']
del df_test['weather_main']


from sklearn.model_selection import KFold
final_model = RandomForestClassifier(n_estimators=23, random_state=1, max_depth = 8)
kfold = KFold(n_splits=8, shuffle=True, random_state=1)
scores = []
for full_train_index, val_index in (kfold.split(df_full_train)):
  full_train = df_full_train.iloc[full_train_index].reset_index(drop=True)
  val = df_full_train.iloc[val_index].reset_index(drop=True)
  target_full_train = full_train.weather_main.values
  target_val = val.weather_main.values
  del full_train['weather_main']
  del val['weather_main']
  X_full_train = dv.fit_transform(full_train.to_dict(orient='records'))
  X_val = dv.transform(val.to_dict(orient='records'))
  final_model.fit(X_full_train, target_full_train)
  y_pred = final_model.predict_proba(X_val).T[1]
  scores.append(roc_auc_score(target_val, y_pred))
print ("Mean value:", np.mean(scores))
X_test = dv.fit_transform(df_test.to_dict(orient='records'))
y_pred = final_model.predict_proba(X_test).T[1]
print("auc on test data",roc_auc_score(target_test, y_pred))


import pickle
output_file = "model.bin"
with open (output_file, "wb") as f_out:
    pickle.dump((dv, final_model), f_out)









