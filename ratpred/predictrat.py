import pandas as pd
import pickle
import json

def predict_ratings(model_path, json_input):
    """
    Predict ratings based on the input features and the model.

    :param model_path: Path to the saved model file.
    :param json_input: JSON string containing the input features.
    :return: Predicted rating.
    """
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Convert JSON input to DataFrame
    features = json.loads(json_input)
    data_point = pd.DataFrame([features])

    # Make a prediction
    prediction = model.predict(data_point)
    return prediction[0]

if __name__ == "__main__":
    json_input = '''
    {
        "Category": 31,
        "Rating Count": 1000,
        "Maximum Installs": 50000,
        "App Age": 12,
        "Size": 15000,
        "Ad Supported": 1,
        "Minimum Android": 8,
        "In App Purchases": 0,
        "Content Rating": 3,
        "Has Developer Website": 1,
        "Price": 0.99,
        "Free": 0,
        "Has Privacy Policy": 1,
        "Minimum Installs": 10000
    }
    '''
    model_path = 'rating_model.pkl'
    prediction = predict_ratings(model_path, json_input)
    print("Predicted Value:", prediction)
