from flask import *
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('classifier.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('frontend.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    input_Values=[np.array(int_features)]
    prediction=model.predict(input_Values)

    if prediction==2:
        return render_template('frontend.html',pred='Employee Work Performance will be low.', x="Performance Rating is 2/4")
    elif prediction==3:
        return render_template('frontend.html', pred='Employee Work Performance will be Medium.', x="Performance Rating is 3/4")
    if prediction==4:
        return render_template('frontend.html', pred='Employee Work Performance will be High.', x="Performance Rating is 4/4")
    else:
        return render_template('frontend.html', pred='Invalid Prediction', x="Train the model again")

if __name__ == '__main__':
    app.run(debug=True)
 