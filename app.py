import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import subprocess, re, pandas as pd, json

# —— 全局日志配置 —— #
LOG_FILE = os.path.join(os.path.dirname(__file__), 'app.log')
logger = logging.getLogger('my_flask_app')
logger.setLevel(logging.DEBUG)
rotating_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
rotating_handler.setFormatter(formatter)
logger.addHandler(rotating_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# —— Flask 应用 —— #
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
app.logger = logger

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, 'predictions.csv')

@app.route('/api/predict_stream', methods=['GET'])
def predict_stream():
    def generate():
        app.logger.debug(">>> 使用的 predictions.csv 路径: %s", CSV_PATH)
        # 1. 如果已有 CSV，直接返回 done 事件，不跑脚本
        if os.path.exists(CSV_PATH):
            app.logger.debug("检测到已存在 %s，直接返回结果，不重复跑脚本", CSV_PATH)
            records = pd.read_csv(CSV_PATH).to_dict(orient='records')
            yield f"event: done\ndata: {json.dumps(records, ensure_ascii=False)}\n\n"
            return

        # 2. 否则启动 predict.py，实时推送进度
        app.logger.debug("未检测到 %s，开始执行 predict.py", CSV_PATH)
        proc = subprocess.Popen(
            ['python', '-u', os.path.join(BASE_DIR, 'predict.py')],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            app.logger.debug("predict.py 输出：%s", line.strip())
            m = re.search(r'Embedding:\s*(\d+)%', line)
            if m:
                yield f"event: progress\ndata: {m.group(1)}\n\n"
        proc.wait()

        # 3. 脚本执行完，再读 CSV 并发送 done
        try:
            records = pd.read_csv(CSV_PATH).to_dict(orient='records')
        except Exception as e:
            app.logger.error("读取 %s 失败：%s", CSV_PATH, e)
            yield f"event: error\ndata: 读取预测结果失败\n\n"
            return

        yield f"event: done\ndata: {json.dumps(records, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
