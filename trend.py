import numpy as np
from config import TREND_THRESHOLD

def detect_trend(anomaly, predicted_returns):
    """Gets data about the anomaly and the predicted returns from the model
        
        Returns 1 if i thinks the anomaly will develop into a trend and 0 otherwise"""
    cumulative_log_return = np.sum(predicted_returns)
    prediction_direction = int(np.sign(cumulative_log_return))
    anomaly_direction = int(anomaly.direction)

    if anomaly_direction != prediction_direction: # If anomaly and prediction are not in the same direction, stop
        return 0

    anomaly_mean = anomaly.window_mean
    future_mean_return = np.sum(predicted_returns)
    future_standard_error =  np.std(predicted_returns) / np.sqrt(len(predicted_returns))

    future_z_score = (future_mean_return - anomaly_mean) / future_standard_error

    trend_label = int(future_z_score >= TREND_THRESHOLD)
    return trend_label