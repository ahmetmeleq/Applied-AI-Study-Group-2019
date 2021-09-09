#Important Modules
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__,template_folder='templates',static_folder='static:css')

model = joblib.load("./model")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,len(to_predict_list))
    result = model.predict(to_predict)
    return result[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("index.html", prediction_text=prediction))

if __name__ == "__main__":
    app.run(debug=True)

