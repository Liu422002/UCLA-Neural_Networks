import os
import logging

from src.data_loader import load_data
from src.preprocessing import preprocess_data, split_features_target
from src.data_splitter import split_data
from src.scaler import scale_data
from src.train import train_mlp_model
from src.evaluation import evaluate_model
from src.save_model import save_model


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_folders():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def main():
    setup_logging()
    create_folders()

    try:
        logging.info("Project started")

        data = load_data("data/Admission.csv")
        logging.info("Data loaded successfully")

        clean_data = preprocess_data(data)
        logging.info("Data preprocessing completed")

        X, y = split_features_target(clean_data)
        logging.info("Features and target separated")

        X_train, X_test, y_train, y_test = split_data(X, y)
        logging.info("Train-test split completed")

        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        logging.info("Data scaling completed")

        model = train_mlp_model(X_train_scaled, y_train)
        logging.info("Model training completed")

        train_accuracy, test_accuracy, cm = evaluate_model(
            model, X_train_scaled, y_train, X_test_scaled, y_test
        )

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Confusion Matrix:")
        print(cm)

        logging.info(f"Train Accuracy: {train_accuracy}")
        logging.info(f"Test Accuracy: {test_accuracy}")
        logging.info(f"Confusion Matrix: {cm}")

        save_model(model, "models/mlp_model.pkl")
        save_model(scaler, "models/scaler.pkl")
        logging.info("Model and scaler saved successfully")

        print("Training completed successfully.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        print("An error occurred. Check logs/app.log for details.")


if __name__ == "__main__":
    main()