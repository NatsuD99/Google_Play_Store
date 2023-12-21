#%%
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the trained model
model = joblib.load('model.pkl')

def gif(model, input_data):
    """
    Get important features from the model.
    """
    feature_importances = np.array(model.feature_importances_)
    feature_names = np.array(input_data.columns)
    importance_data = sorted(zip(feature_importances, feature_names), reverse=True)
    return importance_data

def mpwfi(processed_data):
    """
    Make a prediction and return important features.
    """
    prediction = model.predict(processed_data)
    important_features = gif(model, processed_data)
    return prediction, important_features


processed_data = pd.read_csv('test_X.csv')
prediction, important_features = mpwfi(processed_data)
true_values = pd.read_csv('test_Y.csv')
true_values = true_values.squeeze()
rmse = np.sqrt(mean_squared_error(true_values, prediction))

print("Predicted Values:", prediction)
print("Important Features:", important_features)
print("RMSE:", rmse)
# %%
