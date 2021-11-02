import paddle

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == paddle.max(GT)
    corr = paddle.sum(SR == GT)
    tensor_size = SR.shape[0] * SR.shape[1]* SR.shape[2] * SR.shape[3]
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == paddle.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = paddle.sum((SR == 1) & (GT == 1))
    FN = paddle.sum((SR == 0) & (GT == 1))

    SE = float(paddle.sum(TP)) / (float(paddle.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == paddle.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = paddle.sum((SR == 0) & (GT == 0))
    FP = paddle.sum((SR == 1) & (GT == 0))

    SP = float(paddle.sum(TN)) / (float(paddle.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == paddle.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = paddle.sum((SR == 1) & (GT == 1))
    FP = paddle.sum((SR == 1) & (GT == 0))

    PC = float(paddle.sum(TP)) / (float(paddle.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == paddle.max(GT)

    Inter = paddle.sum((SR & GT))
    Union = paddle.sum((SR | GT))

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == paddle.max(GT)

    Inter = paddle.sum((SR & GT))
    DC = float(2 * Inter) / (float(paddle.sum(SR) + paddle.sum(GT)) + 1e-6)

    return DC


