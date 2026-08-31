from flask import Flask, render_template,request
import numpy as np
import joblib
import math

model = joblib.load("model.pkl")
scaler = joblib.load('scaler.pkl')
app = Flask(__name__)

def round_up(value, decimals=0):
  factor = 10**decimals
  return math.ceil(value * factor) / factor


print(round_up(4.121, 2))

@app.route('/',methods=['GET', 'POST'])
def hello_world():
    prediction = None
    if request.method == 'POST':
        irridation=float(request.form.get('IRRADIATION'))
        # EXACT COLUMN MATCH PROTOCOL ENGAGED
        data = [
            float(request.form.get('AMBIENT_TEMPERATURE')),
            irridation,
            float(request.form.get('MODULE_TEMPREATURE'))* irridation,
        ]
        raw_input = np.array(data).reshape(1, -1)
        print(raw_input)

        prediction_arr = model.predict(raw_input)
        prediction = round_up(prediction_arr[0],2)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
