import paddle
import torch
import numpy as np


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == paddle.max(GT)
    corr = paddle.sum(SR == GT)
    tensor_size = SR.shape[0] * SR.shape[1] #* SR.shape[2] * SR.shape[3]
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


def get_sensitivity_1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative

    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP) + torch.sum(FN)) + 1e-6)
    # TP = ((SR == 1) + (GT == 1)) == 2
    # FN = ((SR == 0) + (GT == 1)) == 2
    #
    # SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_accuracy_1(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    # tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    tensor_size = SR.size(0) * SR.size(1)
    acc = float(corr) / float(tensor_size)

    return acc

def get_specificity_1(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN) + torch.sum(FP)) + 1e-6)

    return SP


def get_precision_1(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP) + torch.sum(FP)) + 1e-6)

    return PC


def get_F1_1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity_1(SR, GT, threshold=threshold)
    PC = get_precision_1(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS_1(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte() + GT.byte()) == 2)
    Union = torch.sum((SR.byte() + GT.byte()) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC_1(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte() + GT.byte()) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC

if __name__ == '__main__':
    a=[[0,1,2,3],[0.1,0.8,0.1,0.6]]
    b=[[0,1,0,1],[0,1,0,0]]
    print(get_sensitivity_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))
    print(get_accuracy_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))
    print(get_specificity_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))
    print(get_precision_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))
    print(get_F1_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))
    print(get_JS_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))
    print(get_DC_1(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b))))

    print('=======')
    print(get_sensitivity(paddle.to_tensor(a),paddle.to_tensor(b)))
    print(get_accuracy(paddle.to_tensor(a), paddle.to_tensor(b)))
    print(get_specificity(paddle.to_tensor(a), paddle.to_tensor(b)))
    print(get_precision(paddle.to_tensor(a), paddle.to_tensor(b)))
    print(get_F1(paddle.to_tensor(a), paddle.to_tensor(b)))
    print(get_JS(paddle.to_tensor(a), paddle.to_tensor(b)))
    print(get_DC(paddle.to_tensor(a), paddle.to_tensor(b)))
