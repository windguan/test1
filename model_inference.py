import os
import sys
import numpy as np
import torch
import joblib
import json


# ================= 定义 ManualStandardScaler 类 =================
class ManualStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ================= 添加 model 文件夹路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# 导入模型
try:
    from models import PPIPredictor

    print("Successfully imported PPIPredictor model")
except ImportError as e:
    print(f"Import model failed: {e}")
    sys.exit(1)

# ================= 🔥 关键修改：直接在 Python 里固定配置 =================
CONFIG = {
    "gcn_model_path": r"E:\pythoncode\PPI\saved_models\best_gcn_model.pth",
    "gat_model_path": r"E:\pythoncode\PPI\saved_models\best_gat_model.pth",
    "scaler_path": r"E:\pythoncode\PPI\processed_data\scaler.pkl",
    "protein_features_path": r"E:\pythoncode\PPI\processed_data\protein_features.npy",
    "id_mapping_path": r"E:\pythoncode\PPI\processed_data\id_mapping.npy"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 加载模型和数据 =================
def load_resources(model_type):
    scaler = joblib.load(CONFIG['scaler_path'])
    protein_features = np.load(CONFIG['protein_features_path'], allow_pickle=True).item()

    model = PPIPredictor(input_dim=256, hidden_dim=128, model_type=model_type).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG[f'{model_type}_model_path'], map_location=DEVICE, weights_only=False))
    model.eval()

    return model, scaler, protein_features


# ================= 预测函数 =================
def predict(protein1_id, protein2_id, model_type):
    try:
        model, scaler, protein_features = load_resources(model_type)

        feat1 = protein_features[protein1_id]
        feat2 = protein_features[protein2_id]
        combined_feat = np.concatenate([feat1, feat2]).reshape(1, -1)

        scaled_feat = scaler.transform(combined_feat)

        with torch.no_grad():
            inputs = torch.tensor(scaled_feat, dtype=torch.float32).to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        result = predicted.item()
        confidence_score = confidence.item()

        return {
            'success': True,
            'result': result,
            'confidence': confidence_score,
            'message': 'Prediction successful'
        }

    except KeyError as e:
        return {
            'success': False,
            'message': f'Protein ID not found: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Prediction error: {str(e)}'
        }


# ================= 主入口（简化版，只传3个参数） =================
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Debug mode")
        features = np.load(CONFIG['protein_features_path'], allow_pickle=True).item()
        test_ids = list(features.keys())[:2]
        result = predict(test_ids[0], test_ids[1], 'gcn')
        print(json.dumps(result))
    else:
        # 🔥 现在只需要传3个参数：蛋白1、蛋白2、模型类型
        protein1_id = sys.argv[1]
        protein2_id = sys.argv[2]
        model_type = sys.argv[3]

        result = predict(protein1_id, protein2_id, model_type)
        print(json.dumps(result))