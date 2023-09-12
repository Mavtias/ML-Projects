import pickle
import numpy as np

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    return y_pred[0]

with open('Classification model\churn-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



customer = {
    'customerid':'83844-dfsdf',
    'gender': 'female',
    'seniorcitizen':0,
    'partner':'yes',
    'dependents':'yes',
    'tenure': 10,
    'phoneservice':'no',
    'multiplelines':'no',
    'internetservice':'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'no',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling':'yes',
    'paymentmethod':'electronic_check',
    'monthlycharges':  400,
    'totalcharges': 500    
}

prediction = predict_single(customer, dv, model)

print('prediction: %.3f' % prediction) 

if prediction >= 0.5:
    print('veredict: Churn')
else:
    print('veredict: Not churn')