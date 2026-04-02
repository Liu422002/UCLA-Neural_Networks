from sklearn.model_selection import train_test_split
import logging

def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=123,
            stratify=y
        )
        logging.info("Train-test split successful")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in train-test split: {e}")
        raise