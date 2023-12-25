import pandas as pd
import pickle
import json

class InferenceEngine:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the prediction model.
        """
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def predict(self, processed_data):
        """
        Make a prediction using the loaded model.
        """
        return self.model.predict(processed_data)

    def predict_for_single_data_point(self, features):
        """
        Predict for a single data point.
        """
        data_point = pd.DataFrame([features])
        prediction = self.predict(data_point)
        return prediction[0]

    def predict_from_json(self, json_input):
        """
        Predict from a JSON object.
        """
        features = json.loads(json_input)
        return self.predict_for_single_data_point(features)

# if __name__ == "__main__":
#     #TODO: Can make diffent engines for diffent model and can gove output with best RMSE.
#     engine = InferenceEngine('rating_model.pkl') # Path to the model file
#     # #TODO: fetch the features as JSON and pass it directly to engine
#     json_input = '''
#     {
#         "Category": 31,
#         "Rating Count": 1000,
#         "Maximum Installs": 50000,
#         "App Age": 12,
#         "Size": 15000,
#         "Ad Supported": 1,
#         "Minimum Android": 8,
#         "In App Purchases": 0,
#         "Content Rating": 3,
#         "Has Developer Website": 1,
#         "Price": 0.99,
#         "Free": 0,
#         "Has Privacy Policy": 1,
#         "Minimum Installs": 10000,
#         "Editors Choice": 0
#     }
#     '''
#     prediction = engine.predict_from_json(json_input)
#     print("Predicted Value:", prediction)