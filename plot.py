import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as matcolor
from sklearn.metrics import roc_curve, auc
import os
import random


def ROC_classes(test, predict, model_name, data_set_name):
    colors = list(matcolor.CSS4_COLORS.values())
    condition = lambda color: (int(color[1:3], 16) * 0.2126 + int(color[3:5], 16) * 0.7152 + int(color[5:],
                                                                                                 16) * 0.0722) < 200
    colors = list(filter(condition, colors))

    tpr, fpr, roc_auc = roc_scores(test, predict)
    classes = len(roc_auc) - 1
    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=2)

    for i in range(classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.5, .48), color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for ' + model_name + "-" + data_set_name)
    plt.legend(loc="lower right")
    path = os.path.join(os.getcwd(), data_set_name + "_" + model_name + ".png")
    plt.savefig(path)
    print("figure saved at:", path)
    # plt.show()


def ROC_Algo(scores, data_set_name):
    colors = list(matcolor.CSS4_COLORS.values())
    condition = lambda color: (int(color[1:3], 16) * 0.2126 + int(color[3:5], 16) * 0.7152 + int(color[5:],
                                                                                                 16) * 0.0722) < 200
    colors = list(filter(condition, colors))
    lw = 2
    plt.figure(figsize=(8, 5))

    for algo in scores.keys():
        tpr, fpr, roc_auc = scores[algo]
        plt.plot(fpr, tpr, color=random.choice(colors), lw=lw,
                 label='{0} (auc = {1:0.2f})'
                       ''.format(algo, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.5, .48), color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('macro-average ROC curve for ' + data_set_name)
    plt.legend(loc="lower right")

    path = os.path.join(os.getcwd(), data_set_name + ".png")
    print("figures saved at:", path)
    plt.savefig(path)
    # plt.show()


def roc_scores(test, predict):
    # t1 = sum(x == 0 for x in predict - test) / len(predict)
    classes = len(set(predict))
    ### MACRO
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test))[:, i], np.array(pd.get_dummies(predict))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return tpr, fpr, roc_auc


def plot_from_csv(path):
    file = pd.read_csv(path, index_col=0)
    data_name = os.path.basename(path)
    data_name = data_name[:data_name.rfind(".")]
    y_test = file["real"]
    macro_scores = dict()
    for col in file.columns[1:]:
        pred = file[col]
        ROC_classes(y_test, pred, col, data_name)
        tpr_macro, fpr_macro, roc_auc_macro = roc_scores(y_test, pred)
        macro_scores[col] = [tpr_macro["macro"], fpr_macro["macro"], float(roc_auc_macro["macro"])]
    ROC_Algo(macro_scores, data_name)


plot_from_csv("csv/MSRC.csv")
plot_from_csv("csv/VOC.csv")
plot_from_csv("csv/PIE.csv")
