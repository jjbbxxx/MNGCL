import json
import os
import pandas as pd
import re
import subprocess
from datetime import datetime
from functools import wraps

from flask import Flask, Response, stream_with_context, request, jsonify, send_file, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask_login import login_required
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# —— Flask 应用 & 配置 —— #
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-this-secret')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# CORS 允许前端跨域携带凭证
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:8080"], "methods": ["GET", "POST", "PUT", "OPTIONS"], "allow_headers": "*"}}, supports_credentials=True)

# —— 扩展初始化 —— #
db = SQLAlchemy(app)
login_manager = LoginManager(app)

# —— 用户模型 —— #
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # 新增

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            return jsonify({'error': '权限不足'}), 403
        return f(*args, **kwargs)
    return decorated

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    csv_path = db.Column(db.String(256), nullable=False)     # 保存文件的服务器路径
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # 保存时间

    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# —— 认证接口 —— #
@app.route('/api/register', methods=['POST', 'OPTIONS'])
def api_register():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': '用户名和密码均为必填'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': '用户名已存在'}), 400
    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({'message': '注册并登录成功'})

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def api_login():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': '用户名和密码均为必填'}), 400
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({'message': '登录成功'})
    return jsonify({'error': '用户名或密码错误'}), 401

@app.route('/api/logout', methods=['POST', 'OPTIONS'])
def api_logout():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    logout_user()
    return jsonify({'message': '已登出'})

@app.route('/api/check_auth', methods=['GET'])
def api_check_auth():
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'username': current_user.username,
            'is_admin': current_user.is_admin
        })
    else:
        return jsonify({
            'logged_in': False,
            'username': None,
            'is_admin': False
        })


# —— 预测接口 —— #
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, 'predictions.csv')


@app.route('/api/predict_stream', methods=['GET'])
def predict_stream():
    def generate():
        # 如果已有 CSV，则直接返回结果（并记录一次，但避免重复写入）
        if os.path.exists(CSV_PATH):
            # 记录预测（仅当未记录时）
            _maybe_record_prediction()
            records = pd.read_csv(CSV_PATH, encoding='utf-8').to_dict(orient='records')
            yield f"event: done\ndata: {json.dumps(records, ensure_ascii=False)}\n\n"
            return

        # 否则启动 predict.py 并推送进度
        proc = subprocess.Popen(
            ['python', '-u', os.path.join(BASE_DIR, 'predict.py')],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            m = re.search(r'Embedding:\s*(\d+)%', line)
            if m:
                yield f"event: progress\ndata: {m.group(1)}\n\n"
        proc.wait()

        # 读取 CSV 并返回，同时写入预测记录
        try:
            records = pd.read_csv(CSV_PATH, encoding='utf-8').to_dict(orient='records')
        except Exception:
            yield f"event: error\ndata: 读取预测结果失败\n\n"
            return

        # 写入 Prediction 表
        _maybe_record_prediction()

        yield f"event: done\ndata: {json.dumps(records, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


def _maybe_record_prediction():
    """如果当前用户未对这个 CSV 路径写过记录，就写一条预测记录。"""
    # 当前用户必须已登录
    if not current_user.is_authenticated:
        return

    # 检查是否已有相同文件的记录，避免重复写入
    exists = Prediction.query.filter_by(user_id=current_user.id, csv_path=CSV_PATH).first()
    if exists:
        return

    pred = Prediction(
        user_id=current_user.id,
        csv_path=CSV_PATH
    )
    db.session.add(pred)
    db.session.commit()


# 新增接口：允许用户下载预测结果的 CSV 文件
@app.route('/api/download_csv', methods=['GET'])
def download_csv():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH, as_attachment=True, download_name="predictions.csv")
    else:
        return jsonify({"error": "CSV 文件未找到"}), 404

@app.route('/api/download_csv/<int:prediction_id>', methods=['GET'])
def download_prediction_csv(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    # 只有管理员或本人可以下载
    if not current_user.is_authenticated or (not current_user.is_admin and current_user.id != pred.user_id):
        return jsonify({'error': '权限不足'}), 403
    return send_file(pred.csv_path, as_attachment=True, download_name=f'prediction_{prediction_id}.csv')


@app.route('/api/users', methods=['GET'])
@admin_required
def api_list_users():
    users = User.query.all()
    data = [
        {'id': u.id, 'username': u.username, 'is_admin': u.is_admin}
        for u in users
    ]
    return jsonify(data)

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@admin_required
def api_update_user(user_id):
    u = User.query.get_or_404(user_id)
    data = request.get_json() or {}
    if 'is_admin' in data:
        u.is_admin = bool(data['is_admin'])
        db.session.commit()
    return jsonify({'id': u.id, 'is_admin': u.is_admin})

@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json() or {}
    old = data.get('old_password')
    new = data.get('new_password')

    if not old or not new:
        return jsonify({'error': '旧密码和新密码都是必填'}), 400

    user = current_user
    if not user.check_password(old):
        return jsonify({'error': '旧密码错误'}), 400

    user.set_password(new)
    db.session.commit()
    return jsonify({'message': '密码修改成功'})


@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    if not current_user.is_authenticated:
        return jsonify({'error': '未登录'}), 403

    if current_user.is_admin:
        # 如果是管理员，返回所有预测记录，并包含用户名
        predictions = db.session.query(Prediction, User.username).join(User, Prediction.user_id == User.id).all()
    else:
        # 如果是普通用户，返回该用户自己的预测记录
        predictions = db.session.query(Prediction, User.username).join(User, Prediction.user_id == User.id).filter(
            Prediction.user_id == current_user.id).all()

    # 将数据转化为字典并返回
    predictions_data = [
        {
            'id': pred.id,
            'username': username,  # 返回用户名
            'csv_path': pred.csv_path,
            'timestamp': pred.timestamp
        }
        for pred, username in predictions
    ]

    return jsonify(predictions_data)



# —— 启动 —— #
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5050, debug=True)
