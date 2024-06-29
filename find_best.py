import pandas as pd

# 读取CSV文件
df = pd.read_csv('test_results.csv', header=None, names=['col1', 'col2', 'col3'])

# 计算第三列的绝对值
df['abs_col3'] = df['col3'].abs()

# 找到第三列绝对值最小的10个数，并按从小到大排序
smallest_10 = df.nsmallest(10, 'abs_col3')

# 返回对应的第一列的字符串和数值本身
result = smallest_10[['col1', 'col3']]

# 输出结果
print(result)