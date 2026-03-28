import os
import sys
import json


# ================= 先定义 ManualStandardScaler 类 =================
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


# ================= 后面是原来的代码 =================
# ...

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# 1. 测试导入模型
print("1. 正在测试模型导入...")
try:
    from models import PPIPredictor

    print("   ✅ 模型导入成功")
except Exception as e:
    print(f"   ❌ 模型导入失败: {e}")
    sys.exit(1)

# 2. 测试加载数据文件
print("\n2. 正在测试数据文件...")
import numpy as np
import joblib

try:
    # 改成你的路径
    scaler_path = r"E:\pythoncode\PPI\processed_data\scaler.pkl"
    features_path = r"E:\pythoncode\PPI\processed_data\protein_features.npy"

    scaler = joblib.load(scaler_path)
    print("   ✅ Scaler 加载成功")

    features = np.load(features_path, allow_pickle=True).item()
    print(f"   ✅ 蛋白质特征加载成功 (共 {len(features)} 个蛋白)")

    # 拿两个蛋白ID做测试
    protein_ids = list(features.keys())[:2]
    print(f"   测试用蛋白ID: {protein_ids[0]}, {protein_ids[1]}")

except Exception as e:
    print(f"   ❌ 数据加载失败: {e}")
    sys.exit(1)

# 3. 测试 model_inference.py
print("\n3. 正在测试推理脚本...")
try:
    from model_inference import predict

    # 构建配置
    config = {
        "gcn_model_path": r"E:\pythoncode\PPI\saved_models\best_gcn_model.pth",
        "gat_model_path": r"E:\pythoncode\PPI\saved_models\best_gat_model.pth",
        "scaler_path": scaler_path,
        "protein_features_path": features_path,
        "id_mapping_path": r"E:\pythoncode\PPI\processed_data\id_mapping.npy"
    }

    # 执行预测
    result = predict(protein_ids[0], protein_ids[1], 'gcn', config)

    if result['success']:
        print(f"   ✅ 预测成功!")
        print(f"   结果: {'互作' if result['result'] == 1 else '不互作'}")
        print(f"   置信度: {result['confidence']:.4f}")
    else:
        print(f"   ❌ 预测失败: {result['message']}")

except Exception as e:
    print(f"   ❌ 推理测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n🎉 测试完成!")