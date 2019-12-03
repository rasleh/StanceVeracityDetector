import numpy as np
import sklearn.metrics as sk


class VeracityMajorityBaseline:
    """Class for performing majority voting on veracity data"""
    def __init__(self):
        self.prediction = 0

    def fit(self, data):
        output_values = [row[0] for row in data]
        self.prediction = max(set(output_values), key=output_values.count)

    def predict(self, data):
        return [self.prediction for i in range(len(data))]

    def test(self, data, unverified_cast):
        predicted_labels = self.predict(data)
        actual_labels = [x[0] for x in data]
        if unverified_cast is not 'none':
            c_matrix = sk.confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])
        else:
            c_matrix = sk.confusion_matrix(actual_labels, predicted_labels, labels=[0, 1, 2])
        cm = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()
        acc = sk.accuracy_score(actual_labels, predicted_labels)
        f1_macro = sk.f1_score(actual_labels, predicted_labels, average='macro')
        f1_micro = sk.f1_score(actual_labels, predicted_labels, average='micro')
        return class_acc, acc, f1_macro, f1_micro
