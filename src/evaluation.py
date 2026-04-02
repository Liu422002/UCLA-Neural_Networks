from sklearn.metrics import accuracy_score, confusion_matrix
import logging

def evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)

        logging.info("Model evaluation completed")
        return train_accuracy, test_accuracy, cm
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        raise