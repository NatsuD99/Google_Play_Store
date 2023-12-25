from flask import Flask, request, render_template
import json
from predictrat import predict_ratings

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            "Category": request.form['Category'],
            "Rating Count": request.form['Rating Count'],
            "Maximum Installs": request.form['Maximum Installs'],
            "App Age": request.form['App Age'],
            "Size": request.form['Size'],
            "Ad Supported": request.form['Ad Supported'],
            "Minimum Android": request.form['Minimum Android'],
            "In App Purchases": request.form['In App Purchases'],
            "Content Rating": request.form['Content Rating'],
            "Has Developer Website": request.form['Has Developer Website'],
            "Price": request.form['Price'],
            "Free": request.form['Free'],
            "Has Privacy Policy": request.form['Has Privacy Policy'],
            "Minimum Installs": request.form['Minimum Installs']
        }
        # for key in features:
        #     features[key] = int(features[key])
        for key, value in features.items():
            print(f"{key}: {value} (Type: {type(value)})")
        json_input = json.dumps(features)
        model_path = 'rating_model.pkl'
        prediction = predict_ratings(model_path, json_input)
        output = round(prediction, 2)

        return render_template('index.html', prediction_text='Predicted App Score: {}'.format(output))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)
