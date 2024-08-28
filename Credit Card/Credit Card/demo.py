import pickle

# Path to your model
MODEL_PATH = 'model.pkl'

# Load the model
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

# Verify model type
print(f"Model type: {type(model)}")
