from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    import subprocess
    result = subprocess.run(['python', 'predict.py'])
    if result.returncode != 0:
        return jsonify({'error': 'Prediction script failed'}), 500

    try:
        result_df = pd.read_csv('predictions.csv')
    except Exception as e:
        return jsonify({'error': f'Failed to read predictions.csv: {str(e)}'}), 500

    return jsonify(result_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)