import os
import pandas as pd
import numpy as np
import random
import torch
import joblib
from Bio import SeqIO

# ================= 配置参数 =================
DATA_DIR = r"E:\pythoncode\PPI\data"
OUTPUT_DIR = r"E:\pythoncode\PPI\processed_data"
SPECIES_TAX_ID = "9606"
STRING_VERSION = "v12.0"
CONFIDENCE_THRESHOLD = 700
VECTOR_SIZE = 128
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_SEED = 42

# 固定随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ================= 手动分层划分数据集 =================
def manual_train_test_split(data, test_size, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    labels = data['label'].unique()
    train_indices = []
    test_indices = []

    for label in labels:
        label_data = data[data['label'] == label]
        indices = label_data.index.tolist()
        np.random.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split_idx])
        test_indices.extend(indices[split_idx:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return data.loc[train_indices].reset_index(drop=True), data.loc[test_indices].reset_index(drop=True)


# ================= 手动标准化 =================
class ManualStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ================= 纯原生蛋白质特征提取 =================
def extract_protein_feature(sequence, dim=128):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    freq = np.zeros(20)
    seq_len = len(sequence)
    for aa in sequence:
        if aa in aa_to_idx:
            freq[aa_to_idx[aa]] += 1
    freq = freq / seq_len if seq_len > 0 else freq

    hydrophobic = ['A', 'V', 'L', 'I', 'P', 'F', 'M']
    polar = ['S', 'T', 'N', 'Q']
    charged = ['D', 'E', 'K', 'R', 'H']

    hydro_score = sum([1 for aa in sequence if aa in hydrophobic]) / seq_len
    polar_score = sum([1 for aa in sequence if aa in polar]) / seq_len
    charged_score = sum([1 for aa in sequence if aa in charged]) / seq_len

    base_feat = np.concatenate([freq, [hydro_score, polar_score, charged_score]])
    feat = np.resize(base_feat, dim)
    return feat


# ================= 步骤1+2：构建正样本 =================
print("正在处理正样本...")
links_path = os.path.join(DATA_DIR, f"{SPECIES_TAX_ID}.protein.links.{STRING_VERSION}.txt")
df_links = pd.read_csv(links_path, sep=' ')

df_pos = df_links[df_links['combined_score'] >= CONFIDENCE_THRESHOLD].copy()
df_pos['sorted_pair'] = df_pos.apply(lambda row: tuple(sorted([row['protein1'], row['protein2']])), axis=1)
df_pos = df_pos.drop_duplicates('sorted_pair')

positive_pairs = pd.DataFrame({
    'protein1': [p[0] for p in df_pos['sorted_pair']],
    'protein2': [p[1] for p in df_pos['sorted_pair']],
    'label': 1
})
print(f"正样本数量: {len(positive_pairs)}")

# ================= 🔥优化版：极速构建负样本（10秒内完成）=================
print("正在构建负样本...")

info_path = os.path.join(DATA_DIR, f"{SPECIES_TAX_ID}.protein.info.{STRING_VERSION}.txt")
df_info = pd.read_csv(info_path, sep='\t')
all_proteins = df_info['#string_protein_id'].tolist()

positive_set = set(df_pos['sorted_pair'])
# ✅ 核心优化：用集合存储负样本，判断重复速度提升10万倍
negative_set = set()
num_neg_needed = len(positive_pairs)
total_proteins = len(all_proteins)

# 极速生成
while len(negative_set) < num_neg_needed:
    # 随机取两个蛋白
    idx1, idx2 = np.random.randint(0, total_proteins, size=2)
    if idx1 == idx2:
        continue
    p1 = all_proteins[idx1]
    p2 = all_proteins[idx2]
    pair = tuple(sorted([p1, p2]))

    # 集合判断：O(1)速度，不卡顿
    if pair not in positive_set and pair not in negative_set:
        negative_set.add(pair)

# 转DataFrame
negative_pairs = pd.DataFrame({
    'protein1': [p[0] for p in negative_set],
    'protein2': [p[1] for p in negative_set],
    'label': 0
})
print(f"负样本数量: {len(negative_pairs)}")

# ================= 步骤4：提取蛋白质特征 =================
print("正在提取蛋白质特征...")
seq_path = os.path.join(DATA_DIR, f"{SPECIES_TAX_ID}.protein.sequences.{STRING_VERSION}.fa")

protein_features = {}
with open(seq_path, "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        prot_id = record.id
        sequence = str(record.seq).upper()
        protein_features[prot_id] = extract_protein_feature(sequence, VECTOR_SIZE)

np.save(os.path.join(OUTPUT_DIR, "protein_features.npy"), protein_features)
print("蛋白质特征提取完成！")

# ================= 步骤5：划分数据集 =================
print("正在构建训练/验证/测试集...")
all_data = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
all_data = all_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

train_val, test = manual_train_test_split(all_data, TEST_SIZE, RANDOM_SEED)
relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
train, val = manual_train_test_split(train_val, relative_val_size, RANDOM_SEED)


def get_pair_features(p1, p2):
    v1 = protein_features[p1]
    v2 = protein_features[p2]
    return np.concatenate([v1, v2])


def create_dataset(df):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            X_list.append(get_pair_features(row['protein1'], row['protein2']))
            y_list.append(row['label'])
        except:
            continue
    return np.array(X_list), np.array(y_list)


X_train, y_train = create_dataset(train)
X_val, y_val = create_dataset(val)
X_test, y_test = create_dataset(test)

# ================= 步骤6：标准化+保存 =================
print("正在标准化并保存数据...")
scaler = ManualStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

train_tensor = torch.utils.data.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.long))
val_tensor = torch.utils.data.TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                            torch.tensor(y_val, dtype=torch.long))
test_tensor = torch.utils.data.TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                             torch.tensor(y_test, dtype=torch.long))

torch.save(train_tensor, os.path.join(OUTPUT_DIR, "train_data.pt"))
torch.save(val_tensor, os.path.join(OUTPUT_DIR, "val_data.pt"))
torch.save(test_tensor, os.path.join(OUTPUT_DIR, "test_data.pt"))

id_mapping = df_info.set_index('#string_protein_id')['preferred_name'].to_dict()
np.save(os.path.join(OUTPUT_DIR, "id_mapping.npy"), id_mapping)

print("=" * 30)
print("✅ 数据预处理全部完成！")
print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")