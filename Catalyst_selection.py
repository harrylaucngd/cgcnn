import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = 'test_results.csv'
df = pd.read_csv(file_path, header=None)
catalyst_name = df.iloc[:, 0].tolist()
atom1 = []
atom2 = []

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')

# 获取每个催化剂的原子
for item in catalyst_name:
    if len(item) == 4:
        atom1.append(item[0:2])
        atom2.append(item[2:4])
    elif len(item) == 3:
        if item[0:2] in periodic_table:
            atom1.append(item[0:2])
            atom2.append(item[2:3])
        else:
            atom1.append(item[0:1])
            atom2.append(item[1:3])
    else:
        atom1.append(item[0:1])
        atom2.append(item[1:2])

delta_H_predict = df.iloc[:, 2]

output_df = pd.DataFrame((atom1, atom2, delta_H_predict.T)).T

# 绝对值
delta_H_predict_abs = delta_H_predict.abs()
# 找到第三列中最接近0的5个数据的索引
closest_indices = delta_H_predict_abs.nsmallest(10).index
closest_data = df.iloc[closest_indices, [0, 2]]
result = list(zip(closest_data.iloc[:, 0], closest_data.iloc[:, 1]))
print(result)

# 热图
x = output_df.iloc[:, 0]
y = output_df.iloc[:, 1]
values = delta_H_predict_abs
x_unique = np.sort(x.unique())
y_unique = np.sort(y.unique())
heatmap_data = np.full((len(y_unique), len(x_unique)), np.nan)
for i in range(len(df)):
    x_index = np.where(x_unique == x[i])[0][0]
    y_index = np.where(y_unique == y[i])[0][0]
    heatmap_data[y_index, x_index] = values[i]


plt.figure(figsize=(15, 10))
ax = sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.1,
                 cbar_kws={'label': r'|$\Delta G_H^*$|'}, linecolor='gray', linewidth=0.1)
ax.set_xticks(np.arange(len(x_unique)) + 0.5)
ax.set_yticks(np.arange(len(y_unique)) + 0.5)
ax.set_xticklabels(x_unique)
ax.set_yticklabels(y_unique)
ax.set_xlabel('atom 1')
ax.set_ylabel('atom 2')
ax.set_title('Model prediction of diatomic catalysts')

# 突出显示最接近0的10个点
for idx in closest_indices:
    x_val = x[idx]
    y_val = y[idx]
    x_index = np.where(x_unique == x_val)[0][0]
    y_index = np.where(y_unique == y_val)[0][0]
    ax.add_patch(plt.Rectangle((x_index, y_index), 1, 1, fill=False, edgecolor='red', linewidth=3))

plt.show()

# 柱状图
plt.figure(figsize=(10, 6))
plt.bar(closest_data.iloc[:, 0], closest_data.iloc[:, 1])
plt.xlabel('diatomic catalysts')
plt.ylabel(r'$\Delta G_H^*$')
plt.title('The 5 diatomic catalysts with the closest adsorption energy to 0')
plt.show()