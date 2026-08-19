### CONFIG
# Data
REQUIRED_COLUMNS = ["date", "ticker", "log_return_1d"] # That you need for anomaly + prediction
# Anomaly
SCAN_DATE = "2026-05-01" # Date to scan for anomaly
ANOMALY_BASELINE_WINDOW = 60 # How many days back are considered the norm of the stock 
ANOMALY_MAX_LENGTH = 7 # How many days back the algorithm looks for before it decided there is no anomaly
ANOMALY_THRESHOLD = 0.1 # How much higher a period needs to be in order to be considered an anomaly
# Trend
TREND_LENGTH = [30, 60] # How far does the model try and predict
TREND_THRESHOLD = 0.1 # How high the prediction z score needs to be for the predictions to be considred a trend
# Models
MODEL_SAVE_PATH = "models/trained" # Folder where trained models are saved to
MODEL_NAME = "dummy_naive_algorithm.pt2" # Name of model to use
REQUIRED_MODEL_COLUMNS = ["date", "ticker", "high_adj", "low_adj", "log_return_1d"] # Columns that is required to use the model
MODEL_FEATURE_COLUMNS = ["high_adj", "low_adj", "log_return_1d"] # The features the model is actually trained on 
MODEL_INPUT_SIZE = 60 # How many days back the model uses in order to predict