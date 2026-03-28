import numpy as np
import pandas as pd

# 加载 id_mapping.npy
id_mapping_path = r"E:\pythoncode\PPI\processed_data\id_mapping.npy"
id_mapping = np.load(id_mapping_path, allow_pickle=True).item()

# 转成 DataFrame
df = pd.DataFrame(list(id_mapping.items()), columns=['string_id', 'gene_name'])

# 保存为 CSV
csv_path = r"E:\pythoncode\PPI\processed_data\protein_list.csv"
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"✅ 已生成蛋白质列表 CSV: {csv_path}")
print(f"   共 {len(df)} 个蛋白质")