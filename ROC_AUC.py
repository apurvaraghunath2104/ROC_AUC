import pandas as pd

# trainPath = "3-data-1.txt"
trainPath = "3-data-2.txt"
# trainPath = "3-data-3.txt"
# trainPath = "3-data-4.txt"
# trainPath = "3-data-5.txt"

df = pd.read_csv(trainPath, sep = ",", header = None)
df = df.dropna()


def calssification_accuracy(prediction_list, class_list):
    count = 0
    for i in range(0, len(prediction_list)):
        if class_list[i] == prediction_list[i]:
            count+= 1

    class_accuracy = count/len(prediction_list)
    # print("class_accuracy", class_accuracy)
    # return class_accuracy
    return round(class_accuracy,3)


def positive_precision(num_of_tps, num_of_fps):
    precision_value = (num_of_tps) / (num_of_tps + num_of_fps)
    # print("positive_precision_value", precision_value)
    # return precision_value
    return round(precision_value, 3)

def negative_precision(num_of_tns, num_of_fps):
    precision_value = (num_of_tns) / (num_of_tns + num_of_fps)
    # print("negative_precision_value", precision_value)
    # return precision_value
    return round(precision_value, 3)


def recall(num_of_tps, num_of_fns):
    recall_value = (num_of_tps) / (num_of_tps + num_of_fns)
    # print("recall_value", recall_value)
    # return recall_value
    return round(recall_value, 3)


def f1_score(precision_value, recall_value):
    f1_score_value = (2 * precision_value * recall_value) / (precision_value + recall_value)
    # print("f1_score_value", f1_score_value)
    # return f1_score_value
    return round(f1_score_value, 3)


def false_positive_rate(num_of_fps, num_of_tns):
    fpr_value = num_of_fps / (num_of_fps + num_of_tns)
    # print("false_positive_rate", fpr_value)
    # return fpr_value
    return round(fpr_value, 3)

def specifity(num_of_tns, num_of_fps):
    specifity_value = num_of_tns / (num_of_tns + num_of_fps)
    # print("specifity_value", specifity_value)
    # return specifity_value
    return round(specifity_value, 3)


class_name = df.iloc[:, -1:]
class_list = class_name.values.tolist()
class_list = [j for i in class_list for j in i]
total = len(class_list)
# print("class_list",class_list)

probability = df.iloc[:, 1]
probability_list = probability.values.tolist()
# print("probability_list",probability_list)

pr_rate = 0.5
prediction_list = []
for i in probability_list:
    if i > pr_rate:
        prediction_list.append(1)
    else:
        prediction_list.append(0)
# print("prediction_list", prediction_list)

TFPN_list = []

for i in zip(prediction_list, class_list):
    if i[0] == 0 and i[1] == 1:
        TFPN_list.append("FN")
    elif i[0] == 1 and i[1] == 1:
        TFPN_list.append("TP")
    elif i[0] == 1 and i[1] == 0:
        TFPN_list.append("FP")
    elif i[0] == 0 and i[1] == 0:
        TFPN_list.append("TN")
# print("TFPN_list", TFPN_list)

num_of_tps = TFPN_list.count("TP")
num_of_tns = TFPN_list.count("TN")
num_of_fps = TFPN_list.count("FP")
num_of_fns = TFPN_list.count("FN")
positive_precision_value = positive_precision(num_of_tps, num_of_fps)
negative_class_precision = negative_precision(num_of_tns, num_of_fps)
recall_value = recall(num_of_tps, num_of_fns)
f1_score_value = f1_score(positive_precision_value, recall_value)
true_positive_rate = recall_value
# print("true_positive_rate", true_positive_rate)
false_positive_rate = false_positive_rate(num_of_fps, num_of_tns)
sensitivity = true_positive_rate
# print("sensitivity", sensitivity)
specifity = specifity(num_of_tns, num_of_fps)
class_accuracy = calssification_accuracy(prediction_list, class_list)


fpr_list = []
tpr_list = []
k = []
for i,j in zip(probability_list, class_list):
    k.append((i,j))
k.sort(key=lambda x: x[0])


actual_class = []
sorted_k = sorted(k, reverse = True)
# print("sorted_k", sorted_k)
for i in range(0, len(sorted_k)):
    actual_class.append(sorted_k[i][1])
# print("ACTUAL CLASS", actual_class)


for i in range(0, len(actual_class)):
    temp_list = []
    left_list = actual_class[0:i+1]
    num_of_tps = left_list.count(1)
    num_of_fps = left_list.count(0)
    right_list = actual_class[i+1:]
    num_of_tns = right_list.count(0)
    num_of_fns = right_list.count(1)
    temp_list.append(num_of_tps)
    temp_list.append(num_of_fps)
    temp_list.append(num_of_tns)
    temp_list.append(num_of_fns)
    fpr_value = num_of_fps / (num_of_fps + num_of_tns)
    fpr_list.append(fpr_value)
    tpr_value = num_of_tps / (num_of_tps + num_of_fns)
    tpr_list.append(tpr_value)

# print("TPR_LIST", tpr_list)
# print("FPR_LIST", fpr_list)
auc = 0
for i in range(1, len(tpr_list)):
    h = fpr_list[i] - fpr_list[i - 1]
    auc += h * (tpr_list[i - 1] + tpr_list[i]) / 2
# print("AUC", round(auc,3))

with open("3_result.txt","w") as fd:
    fd.write("(63\n" + "\n((Accuracy " + str(class_accuracy)+ ")\n\n")
    fd.write("(Precision " + str(positive_precision_value) + ")\n\n")
    fd.write("(Recall " + str(recall_value) + ")\n\n")
    fd.write("(F1 " + str(f1_score_value) + ")\n\n")
    fd.write("(TPR " + str(true_positive_rate) + ")\n")
    fd.write("(FPR " + str(false_positive_rate) + ")\n\n")
    fd.write("(Specificity " + str(specifity) + ")\n\n")
    fd.write("(Sensitivity " + str(sensitivity) + ")\n\n")
    fd.write("(AUC " + str(round(auc,3)) + "))\n\n")
    fd.write(")\n")
