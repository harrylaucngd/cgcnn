import pandas as pd
import os

file_path = "./DACs-data-predict/Unlabeled_data.csv"

df = pd.read_csv(file_path, header=None)

# 删除第一列中的 ',cif' 字符
df[0] = df[0].str.replace('.cif', '')
df.to_csv(file_path, index=False, header=False)