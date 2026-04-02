import pickle
import logging

def save_model(model, file_path):
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise