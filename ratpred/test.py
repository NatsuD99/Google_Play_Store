import pickle
import numpy as np

with open('rating_model.pkl', 'rb') as f:
    model = pickle.load(f)

sample_input = np.array([[31, 1000, 50000, 12, 15000, 1, 8, 0, 3, 1, 0.99, 0, 1, 10000]])
print(model.predict(sample_input))
