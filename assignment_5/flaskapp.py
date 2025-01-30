from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
from cus_p_r import CustomPerceptronDotProduct

app = Flask(__name__)
CORS(app)  

# Mean and standard deviation values for feature scaling
mean = [120.032717, 64.002845, 32.137411, 32.897582]
std = [29.714315, 79.664414, 6.498238, 11.063700]

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract features from the data and convert to float
    glucose = float(data.get("glucose"))
    insulin = float(data.get("insulin"))
    bmi = float(data.get("bmi"))
    age = float(data.get("age"))
    model_choice = data.get("model")  # Get the selected model from the frontend
    
    if None in [glucose, insulin, bmi, age, model_choice]:
        return jsonify({"error": "Missing features or model selection in the request"}), 400

    # Prepare the input for prediction
    features = [glucose, insulin, bmi, age]

    # Scale the features
    scaled_features = [(features[i] - mean[i]) / std[i] for i in range(len(features))]

    # Load the appropriate model based on user choice
    if model_choice == "perceptron":
        model = pickle.load(open('perceptron_modelR.pkl', 'rb'))
    elif model_choice == "naive_bayes":
        model = pickle.load(open('naive_bayes_modelR.pkl', 'rb'))
    else:
        return jsonify({"error": "Invalid model choice"}), 400
    
    # Make a prediction
    prediction = model.predict([scaled_features])
    
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
