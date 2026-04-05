from flask import Flask,render_template,jsonify,request
import pickle,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

model=pickle.load(open('ridge.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/prediction",methods=['GET','POST'])
def pred_datapoint():
    if request.method=='POST':
        Temprature=float(request.form.get('Temprature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_scaled_data=scaler.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=model.predict(new_scaled_data)
        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')
    

if __name__=="__main__":
    app.run(debug=True)

