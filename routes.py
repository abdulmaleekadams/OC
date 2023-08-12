from flask import request, jsonify, Blueprint
import pandas as pd
from ml_model import xgb_model

main_bp = Blueprint('main', __name__)

@main_bp.route("/api/index", methods=['POST'])
def hello_world():
    data = request.json

    # Extract user input data
    age = float(data['age'])
    cea = float(data['cea'])
    ibil = float(data['ibil'])
    neu = float(data['neu'])
    menopause = float(data['menopause'])
    ca125 = float(data['ca125'])
    alb = float(data['alb'])
    he4 = float(data['he4'])
    glo = float(data['glo'])
    lym = float(data['lym%'])
    
    # Create a new DataFrame with user input
    user_data = pd.DataFrame([[age, cea, ibil, neu, menopause, ca125, alb, he4, glo, lym]], columns=[
                             'Age', 'CEA', 'IBIL', 'NEU', 'Menopause', 'CA125', 'ALB', 'HE4', 'GLO', 'LYM%'])
    
    # Make predictions using the trained model
    predictions = xgb_model.predict(user_data)

    # Interpret the predictions
    if predictions == 1:
        return jsonify({'result': "The model predicts the absence of cancer."})
    else:
        return jsonify({'result': "The model predicts the presence of cancer."})
