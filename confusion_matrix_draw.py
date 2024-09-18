import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 定义所有分类标签
allLabels = ['AM', 'FM', 'COMB', 'ISRI', 'SMSP', 'NP']

# 定义部分混淆矩阵数据
partial_confMatrix = np.array([
    [  1166 ,  21 ,  10  ,  3],
    [   13 , 839  ,332  , 16],
    [   19 , 807 , 359 ,  15],
    [  0  , 35 ,  31, 1134]
])

# 部分数据对应的标签
# partial_labels = ['FM', 'COMB', 'SMSP', 'NP']
# partial_labels = ['AM', 'FM', 'SMSP', 'NP']
# partial_labels = ['FM', 'COMB', 'ISRI', 'SMSP']
partial_labels = ['AM', 'COMB', 'ISRI', 'NP']



# 创建一个全零的矩阵，大小为 len(allLabels) x len(allLabels)
num_labels = len(allLabels)
full_confMatrix = np.zeros((num_labels, num_labels), dtype=int)

# 填充部分混淆矩阵数据到完整的混淆矩阵中
for i, row_label in enumerate(partial_labels):
    row_index = allLabels.index(row_label)
    for j, col_label in enumerate(partial_labels):
        col_index = allLabels.index(col_label)
        full_confMatrix[row_index, col_index] = partial_confMatrix[i, j]

# 调整图形大小
fig, ax = plt.subplots(figsize=(10, 8))  # 增加图形大小

# 使用 ConfusionMatrixDisplay 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=full_confMatrix, display_labels=allLabels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)  # 旋转 x 轴标签

# 添加标题和标签
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图形
plt.show()
