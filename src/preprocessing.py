import pandas as pd
import logging

def preprocess_data(data):
    try:
        data["Admit_Chance"] = (data["Admit_Chance"] > 0.8).astype(int)

        if "Serial_No" in data.columns:
            data = data.drop(["Serial_No"], axis=1)

        data["University_Rating"] = data["University_Rating"].astype("object")
        data["Research"] = data["Research"].astype("object")

        clean_data = pd.get_dummies(
            data,
            columns=["University_Rating", "Research"],
            drop_first=True,
            dtype=int
        )

        logging.info("Preprocessing completed successfully")
        return clean_data

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise


def split_features_target(data):
    try:
        X = data.drop(["Admit_Chance"], axis=1)
        y = data["Admit_Chance"]
        return X, y
    except Exception as e:
        logging.error(f"Error splitting features and target: {e}")
        raise