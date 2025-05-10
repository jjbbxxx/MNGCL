from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os
from flask import Response, stream_with_context
import subprocess, re
import pandas as pd
import json

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/api/predict', methods=['GET'])
def predict():
    result = subprocess.run(['python', 'predict.py'])
    if result.returncode != 0:
        return jsonify({'error': 'Prediction failed'}), 500
    df = pd.read_csv('predictions.csv')
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/predict_stream', methods=['GET'])
def predict_stream():
    def generate():
        # 启动本地脚本并捕获 stdout
        proc = subprocess.Popen(
            ['python', '-u', 'predict.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        # 实时读取每一行
        for line in proc.stdout:
            # 假设脚本里有 "Embedding: xx%" 的打印
            m = re.search(r'Embedding:\s*(\d+)%', line)
            if m:
                yield f"event: progress\ndata: {m.group(1)}\n\n"
        proc.wait()
        # 读取最终生成的 CSV
        records = pd.read_csv('predictions.csv').to_dict(orient='records')
        json_data = json.dumps(records, ensure_ascii=False)
        yield f"event: done\ndata: {json_data}\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)