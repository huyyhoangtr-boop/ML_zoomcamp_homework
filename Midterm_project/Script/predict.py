
import numpy as np
import pickle
import pandas as pd


output_file = "model.bin"


with open(output_file,"rb") as f_in:
    dv, model = pickle.load (f_in)
from flask import Flask 
from flask import request, jsonify
app = Flask("predict")

@app.route("/predict",methods= ["POST"])








    
def predict():
    data  =request.get_json()
   
    X_train = dv.transform(data)
    y_pred = model.predict_proba(X_train)
    result = {
        "rain prob": float(list(y_pred.T[1])[0]),
        "rain": bool(y_pred.T[1] >=0.5)
    }


    
    return jsonify(result)


if __name__ =="__main__":
    app.run(debug=True, host = "0.0.0.0", port =9000)

