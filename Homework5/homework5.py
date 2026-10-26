import pickle

output_file = "pipeline.bin"
customer = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
with open(output_file,"rb") as f_in:
    dv, model = pickle.load (f_in)





    
def predict():
    
    #dv.fit(df.to_dict(orient='records')) 
    X_train = dv.transform(customer)
    y_pred = model.predict_proba(X_train)
    result = {
        "churn prob": float(list(y_pred.T[1])[0]),
        "churn": bool(y_pred.T[1] >=0.5)
    }


    
    return (result)

print (predict())
