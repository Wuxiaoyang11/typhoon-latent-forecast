from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# 计算并返回平衡准确率分数
def get_balance_accuracy_score(preds, labels):
    """
    计算预测值和真实标签之间的平衡准确率。
    平衡准确率在处理不平衡数据集时特别有用，它是每个类别召回率的宏平均。

    参数:
        preds (array-like): 模型的预测结果。
        labels (array-like): 真实的标签。

    返回:
        float: 平衡准确率分数。
    """
    return balanced_accuracy_score(labels, preds)

# 打印混淆矩阵
def print_confusion_matrix(preds, labels):
    """
    计算并打印预测值和真实标签之间的混淆矩阵。
    混淆矩阵可以清晰地显示模型在各个类别上的表现。

    参数:
        preds (array-like): 模型的预测结果。
        labels (array-like): 真实的标签。
    """
    print("混淆矩阵:")
    # 使用sklearn的confusion_matrix函数计算混淆矩阵并打印
    print(confusion_matrix(labels, preds))
