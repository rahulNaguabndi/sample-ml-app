import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
app = Flask(__name__)


@app.route('/predict',methods=['POST'])
def predict():
    request_data = request.get_json()
    age = request_data["age"]
    sex= request_data["sex"]
    chestPainTye= request_data["chestPainTye"]
    restingBloodPreassure= request_data["restingBloodPreassure"]
    cholestoral= request_data["cholestoral"]
    fastingBloodSugar= request_data["fastingBloodSugar"]
    restingEcg= request_data["restingEcg"]
    maximumHeartRate= request_data["maximumHeartRate"]
    exerciseInducedAngia= request_data["exerciseInducedAngia"]
    oldPeak= request_data["oldPeak"]
    slp= request_data["slp"]
    caa= request_data["caa"]
    thalassemia= request_data["thalassemia"]
    predction = model.predict(scaler.transform([[age,sex,chestPainTye,restingBloodPreassure,cholestoral,fastingBloodSugar,restingEcg,maximumHeartRate,exerciseInducedAngia,oldPeak,slp,caa,thalassemia]]))
    result  = ""
    if(predction[0] == 0):
        result = "No Risk of Heart Disease"
    else:
        result = "Risk of Heart Disease"
    return result

if __name__ == "__main__":
    app.run(debug=True)