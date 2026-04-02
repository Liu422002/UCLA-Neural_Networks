from sklearn.preprocessing import MinMaxScaler
import logging

def scale_data(X_train, X_test):
    try:
        scaler = MinMaxScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Scaling successful")
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logging.error(f"Error in scaling data: {e}")
        raise