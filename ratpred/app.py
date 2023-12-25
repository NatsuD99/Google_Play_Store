from flask import Flask, request, render_template
import json
from predictrat import predict_ratings

app = Flask(__name__)
categories = [
    'Education',
    'Business',
    'Tools',
    'Entertainment',
    'Lifestyle',
    'Books',
    'Personalization',
    'Health',
    'Productivity',
    'Shopping',
    'Food',
    'Travel',
    'Finance',
    'Arcade',
    'Puzzle',
    'Casual',
    'Communication',
    'Sports',
    'Social',
    'News',
    'Photography',
    'Medical',
    'Action',
    'Maps',
    'Simulation',
    'Adventure',
    'Educational',
    'Art',
    'Auto',
    'House',
    'Video',
    'Events',
    'Beauty',
    'Trivia',
    'Board',
    'Racing',
    'Role',
    'Word',
    'Strategy',
    'Card',
    'Weather',
    'Dating',
    'Libraries',
    'Casino',
    'Music',
    'Parenting',
    'Comics'
]

@app.route('/')
def home():
    return render_template('index.html', categories=categories)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            "Category": request.form['Category'],  #text
            "Rating Count": request.form['Rating Count'],   #number
            "Maximum Installs": request.form['Maximum Installs'], #number
            "App Age": request.form['App Age'],     #number
            "Size": request.form['Size'], #number
            "Ad Supported": request.form['Ad Supported'], #select
            "Minimum Android": request.form['Minimum Android'], #number
            "In App Purchases": request.form['In App Purchases'], #select
            "Content Rating": request.form['Content Rating'], #number
            "Has Developer Website": request.form['Has Developer Website'], #select
            "Price": request.form['Price'], #number
            "Free": request.form['Free'], #select
            "Has Privacy Policy": request.form['Has Privacy Policy'], #select
            "Minimum Installs": request.form['Minimum Installs'] #number
        }
        # for key in features:
        #     features[key] = int(features[key])
        # Note: The values are coming in string format. We need to convert them to the appropriate data types.
        #       For example, the "Category" feature is an integer, but it is coming in as a string.
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
