### CONFIG
REQUIRED_COLUMNS = ["date", "ticker", "log_return_1d"] # That you need for anomaly + prediction
SCAN_DATE = "2026-05-01" # Date to scan for anomaly
ANOMALY_BASELINE_WINDOW = 60 # How many days back are considered the norm of the stock 
ANOMALY_MAX_LENGTH = 7 # How many days back the algorithm looks for before it decided there is no anomaly
ANOMALY_THRESHOLD = 1.5 # How much higher a period needs to be in order to be considered an anomaly
TREND_LENGTH = [30, 60] # How far does the model try and predict
TREND_THRESHOLD = 0.0 # How high the prediction z score needs to be for the predictions to be considred a trend