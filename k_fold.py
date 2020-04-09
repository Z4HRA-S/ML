import numpy as np


def split_data(data, k=5):
    np.random.shuffle(data)
    return np.array_split(data, k)


def score(predicted_label, test_label):
    classes = set(test_label)
    confusion = np.zeros((len(classes), len(classes)))
    for label in zip(predicted_label, test_label):
        predicted = int(label[0] - 1)
        actual = int(label[1] - 1)
        confusion[predicted][actual] += 1

    avg_precision, avg_recall, avg_specificity, avg_f1 = 0, 0, 0, 0
    for c in classes:
        index = int(c - 1)
        support = sum(confusion[:, index])
        tp = confusion[index][index]
        fn = support - tp
        fp = sum(confusion[index]) - tp
        tn = len(test_label) - (tp + fn + fp)
        if support == 0 and fp == 0:
            precision = 1
        elif support != 0 and tp == 0 and fp == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)

        if support == 0 and fn == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)

        avg_precision += precision * support
        avg_recall += recall * support
        avg_specificity += (tn / (tn + fp)) * support
        if precision != 0 or recall != 0:
            avg_f1 += (2 * precision * recall / (precision + recall)) * support

    # calculate weighted average of scores
    avg_precision = avg_precision / len(test_label)
    avg_recall = avg_recall / len(test_label)
    avg_specificity = avg_specificity / len(test_label)
    accuracy = sum([confusion[i][i] for i in range(len(classes))]) / len(test_label)
    avg_f1 = avg_f1 / len(test_label)

    return {"accuracy": accuracy, "precision": avg_precision, "f1": avg_f1,
            "recall": avg_recall, "specificity": avg_specificity}


def k_fold(model, data, model_name="NOT_SET", data_set_name="", k=5, split=True):
    if model_name == "NOT_SET":
        model_name = model.name

    if split:
        data = split_data(data, k)
    scores = []
    for i in range(k):
        train_data = np.concatenate(np.delete(data, i, axis=0))
        train_label = train_data[:, -1]
        train_data = train_data[:, :-1]
        test_data = data[i]
        test_label = test_data[:, -1]
        test_data = test_data[:, :-1]
        print("Running ", model_name, "for ", data_set_name, "in", i, "th fold:")
        model.fit(train_data, train_label)
        predicted = model.predict(test_data)
        result = score(predicted, test_label)
        print(result)
        scores.append(result)
        print("=" * 50)

    score_name = scores[0].keys()
    avg_score = {}
    for s in score_name:
        avg_score[s] = sum([scores[i][s] for i in range(k)]) / k

    avg_score["data"] = data_set_name
    avg_score["model"] = model_name
    return avg_score
