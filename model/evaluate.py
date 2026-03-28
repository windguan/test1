import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 导入模型
from models import PPIPredictor

# ================= 配置参数 =================
PROCESSED_DATA_DIR = r"E:\pythoncode\PPI\processed_data"
MODEL_SAVE_DIR = r"E:\pythoncode\PPI\saved_models"
RESULT_SAVE_DIR = r"E:\pythoncode\PPI\results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(RESULT_SAVE_DIR):
    os.makedirs(RESULT_SAVE_DIR)


# ================= 🔥 纯原生实现：所有性能指标（无sklearn） =================
def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.sum(y_true == y_pred) / len(y_true)


def calculate_precision_recall_f1(y_true, y_pred):
    """计算精确率、召回率、F1分数"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def calculate_roc_auc(y_true, y_scores):
    """计算ROC-AUC"""
    # 排序
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    # 计算TPR和FPR
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)

    # 计算AUC（梯形法则）
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0

    return auc


def calculate_pr_auc(y_true, y_scores):
    """计算PR-AUC"""
    # 排序
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # 计算Precision和Recall
    precisions = []
    recalls = []
    tp = 0
    fp = 0
    fn = np.sum(y_true)

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # 计算AUC（梯形法则）
    auc = 0.0
    for i in range(1, len(recalls)):
        auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0

    return auc


# ================= 加载测试数据 =================
def load_test_data():
    test_data = torch.load(os.path.join(PROCESSED_DATA_DIR, "test_data.pt"))
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    return test_loader


# ================= 模型评估函数 =================
def evaluate_model(model_type):
    print(f"\n🔍 正在评估 {model_type.upper()} 模型...")

    # 加载模型
    model = PPIPredictor(input_dim=256, hidden_dim=128, model_type=model_type).to(DEVICE)
    model_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_type}_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_loader = load_test_data()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类概率
            _, preds = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 转numpy数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # 🔥 用原生函数计算指标
    acc = calculate_accuracy(all_labels, all_preds)
    _, _, f1 = calculate_precision_recall_f1(all_labels, all_preds)
    roc_auc = calculate_roc_auc(all_labels, all_probs)
    pr_auc = calculate_pr_auc(all_labels, all_probs)

    print(f"📈 {model_type.upper()} 模型测试结果:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print(f"   PR-AUC:    {pr_auc:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    return {
        'model': model_type,
        'acc': acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'labels': all_labels,
        'probs': all_probs
    }


# ================= 绘制对比图表 =================
def plot_comparison(results_gcn, results_gat):
    print("\n🎨 正在生成性能对比图表...")

    # 1. 指标对比柱状图
    metrics = ['Accuracy', 'ROC-AUC', 'PR-AUC', 'F1-Score']
    gcn_values = [results_gcn['acc'], results_gcn['roc_auc'], results_gcn['pr_auc'], results_gcn['f1']]
    gat_values = [results_gat['acc'], results_gat['roc_auc'], results_gat['pr_auc'], results_gat['f1']]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, gcn_values, width, label='GCN', color='#1f77b4')
    bars2 = plt.bar(x + width / 2, gat_values, width, label='GAT', color='#ff7f0e')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('GCN vs GAT - Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

    plt.savefig(os.path.join(RESULT_SAVE_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print("   图表已保存至: model_comparison.png")


# ================= 主程序 =================
if __name__ == "__main__":
    print("=" * 50)
    print("   基于图神经网络的PPI预测 - 性能评估")
    print("=" * 50)

    # 评估两个模型
    results_gcn = evaluate_model('gcn')
    results_gat = evaluate_model('gat')

    # 绘制对比图
    plot_comparison(results_gcn, results_gat)

    print("\n✅ 性能评估完成！结果保存在:", RESULT_SAVE_DIR)