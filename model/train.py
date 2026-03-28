import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 导入模型
from models import PPIPredictor

# ================= 配置参数 =================
PROCESSED_DATA_DIR = r"E:\pythoncode\PPI\processed_data"
MODEL_SAVE_DIR = r"E:\pythoncode\PPI\saved_models"
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


# ================= 加载数据 =================
def load_data():
    print("正在加载数据...")
    train_data = torch.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pt"))
    val_data = torch.load(os.path.join(PROCESSED_DATA_DIR, "val_data.pt"))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# ================= 训练函数 =================
def train_model(model_type):
    print(f"\n🚀 开始训练 {model_type.upper()} 模型...")

    # 初始化模型
    model = PPIPredictor(input_dim=256, hidden_dim=128, model_type=model_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = load_data()

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_type}_model.pth")

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_acc = 100 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(
            f"📊 Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 保存最优 {model_type.upper()} 模型 (Val Acc: {best_val_acc:.2f}%)")

    print(f"\n✅ {model_type.upper()} 模型训练完成！最优验证准确率: {best_val_acc:.2f}%")


# ================= 主程序 =================
if __name__ == "__main__":
    print("=" * 50)
    print("   基于图神经网络的PPI预测 - 模型训练")
    print("=" * 50)

    # 先训练 GCN
    train_model('gcn')

    # 再训练 GAT
    train_model('gat')

    print("\n🎉 所有模型训练完成！模型保存在:", MODEL_SAVE_DIR)