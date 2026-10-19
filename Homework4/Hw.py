import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("course_lead_scoring_2026.csv")
print (df.dtypes)

#DATA PREPARATION
def data_fill_missing(df, NA):
    categorical_columns = df.columns[df.dtypes == "object" ]
    numerical_columns = df.columns[df.dtypes != "object"]
    for i in categorical_columns:
        if df[i].isnull().sum() >0: df[i] = df[i].fillna(NA)
    for i in numerical_columns:
        if df[i].isnull().sum() >0: df[i] = df[i].fillna(0)
    return df, categorical_columns, numerical_columns   
def data_split(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    target_full_train = df_full_train["converted"]
    target_train = df_train["converted"]
    target_val = df_val["converted"]
    target_test = df_test["converted"]
    
    
    return df_full_train, df_train, df_val, df_test, target_train, target_val, target_test, target_full_train

#Question 1: ROC AUC feature importance
from sklearn.metrics import roc_auc_score as auc
def q1(df, numerical_columns, y):
    area = {}
    for i in numerical_columns:
        if i != "converted":
            area[i] = float(auc(y_true=y, y_score=df[i]))
            if auc(y_true=y, y_score=df[i]) < 0.5: df[i] = -df[i]
    print (area)
    print ("Q1:", max(area,key=area.get))
    return 

#Question 2: Training the model
from sklearn.feature_extraction import DictVectorizer 
from sklearn.linear_model import LogisticRegression as LgR

def q2(df_train, df_val,target_train, target_val, categorical_columns, numerical_columns,c):
    dv = DictVectorizer(sparse=False)

    X_train = df_train[categorical_columns].to_dict(orient = "records")
    
    X_val = df_val[categorical_columns].to_dict(orient = "records")
    X_train = dv.fit_transform(X_train)
    X_val = dv.transform(X_val)
  

    X_train = np.column_stack([df_train[numerical_columns].values, X_train])
    X_val = np.column_stack([ df_val[numerical_columns].values, X_val])
  
    
    model = LgR(solver='liblinear', C=c, max_iter=1000)
    model.fit(X_train, target_train)
    y_pred = model.predict_proba(X_val)
    print ("AUC:", round(auc( y_true=target_val, y_score=y_pred.T[1]),3))
    return y_pred, auc( y_true=target_val, y_score=y_pred.T[1])
#Question 3: Precision and Recall
def q3_q4(y_pred, y_true):
    precision_list = []
    recall_list = []
    for t in np.linspace(0,1,101):
        data=y_true.to_frame()
   
        data["y_predT"] = (y_pred >=t).astype("int")
        data ["y_predF"] = (y_pred <t).astype("int")
        data["true"] = (data["y_predT"] == y_true).astype("int")
        data["false"] = (data["y_predF"] == y_true).astype("int")
        true = data.groupby(y_true).true.sum().values
        
        false = data.groupby(y_true).false.sum().values
        tp = true[1]
        tn = true[0]
        fp = false[0]
        fn = false[1]        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        precision_list.append(precision)
        recall_list.append(recall)
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    difference = np.absolute(precision_list - recall_list)
    #print(difference)
    for i in range(len(difference[~np.isnan(difference)])):
        if difference[~np.isnan(difference)][i] == difference[~np.isnan(difference)].min(): print("threadholds for min diff=", 0.01*(i))
    #There are the code above cannot apply globally as nan can vary across the array.
    
    F1 = 2*precision_list*recall_list/(precision_list+recall_list)
    #print (F1)
    print("threadholds for F1_max=",np.argmax(F1[~np.isnan(F1)])*0.01)
    
    
    plt.plot(precision_list, color = "blue", label = "precision")
    plt.plot(recall_list, color = "red", label = "recall")
    plt.legend()
    plt.show()
    return 

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
def q5(df_full_train,c):
    AUC = []
    for train_idx, val_idx in (kfold.split(df_full_train)):
        train = df_full_train.iloc[train_idx]
        val = df_full_train.iloc[val_idx]
        y_pred, a = q2(train, val, train["converted"], val["converted"], categorical_columns, numerical_columns, c=c)
        AUC.append(a)
    print(np.std(np.array(AUC)))
    return AUC
def q6(C):
    category ={}
    for c in C:
        print("C=",c)
        AUC = q5(df_full_train, c=c)
        print(np.array(AUC).mean().__round__(3), np.std(np.array(AUC)).__round__(3) )
        print()
        category[c] = [np.array(AUC).mean().__round__(3), np.std(np.array(AUC)).__round__(3)]
    print("BEST c=",max(category, key=category.get))
#OPERATION

df, categorical_columns, numerical_columns = data_fill_missing(df, "NA")
df_full_train, df_train, df_val, df_test, target_train, target_val, target_test, target_full_train = data_split(df)
q1(df_train, numerical_columns, df_train["converted"])


del df_train["converted"]
del df_test["converted"]
del df_val["converted"]

numerical_columns = numerical_columns.drop("converted")
y_pred, a = q2(df_train, df_val,target_train, target_val, categorical_columns, numerical_columns,c=1)
q3_q4(y_pred.T[1], target_val)

q5(df_full_train, c=1)
print()
q6([0.000001, 0.001, 1])