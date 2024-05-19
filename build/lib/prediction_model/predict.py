import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset

classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred ==1, "Y", "N")
    result = {"Predictions":output}
    return result

# def generate_predictions():
#     data = load_dataset(config.TEST_FILE)
#     # data = pd.DataFrame(data_input)
#     pred = classification_pipeline.predict(data[config.FEATURES])
#     output = np.where(pred ==1, "Y", "N")
#     # result = {"Predictions":output}
#     return output

if __name__ == "__main__":
    generate_predictions()