from sklearn.neural_network import MLPClassifier
import logging

def train_mlp_model(X_train, y_train):
    try:
        model = MLPClassifier(
            hidden_layer_sizes=(3,),
            batch_size=50,
            max_iter=200,
            random_state=123
        )
        model.fit(X_train, y_train)
        logging.info("MLP model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise