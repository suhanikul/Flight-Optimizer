# model/test_predict_lgb.py
import joblib
from train_model import predict_from_dict

model = joblib.load('model/flight_price_model.pkl')
encoders = joblib.load('model/label_encoders.pkl')

example = {
    'airline': 'Indigo',
    'from': 'Delhi',
    'to': 'Mumbai',
    'class': 'economy',
    'stops': 'non-stop',
    'dep_block': 'Morning',
    'arr_block': 'Morning',
    'duration': 130,
    'day_of_week': 0,
    'is_weekend': 0
}

print('Prediction (₹):', predict_from_dict(model, encoders, example))
