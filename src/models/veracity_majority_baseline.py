import numpy as np
import sklearn.metrics as sk


class VeracityMajorityBaseline:
    """Class for performing majority voting on veracity data"""
    def __init__(self):
        self.prediction = 0

    def fit(self, data):
        """
        Fit model to best predict given input data based on the most common label.

        :param data: a 2d array, with the label of each row given at column 0
        :return: the fitted majority voter model
        """
        output_values = [row[0] for row in data]
        self.prediction = max(set(output_values), key=output_values.count)
        return self

    def predict(self, data):
        """
        Predicts labels for the given data, based on the majority label found on the data used with the "fit" function

        :param data: a 2d array with a data point at each row
        :return: predicted values; an array of the same length as the given data, populated with the majority class at
        each index
        """
        return [self.prediction for i in range(len(data))]

    def test(self, data, unverified_cast):
        """
        Finds model predictions using the predict() function, and compares results with the actual labels, returning
        these as output. Needs indication of how Unverified rumours have been handled; cast as either True or False,
        or left as unverified to turn the task into 3-class prediction.

        :param data: matrix with the dimensions [number of datapoints][x], where z is arbitrary. The label of each row
        must be given at column 0
        :param unverified_cast: how unverified rumours have been handled; is 'none' if they have not been cast as
        another class, or alternatively 'true' or 'false'
        :return: accuracy for each class, overall accuracy, F1 micro and macro averaged, precision and recall
        """
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
        precision = sk.precision_score(actual_labels, predicted_labels)
        recall = sk.recall_score(actual_labels, predicted_labels)
        print("Confusion matrix:")
        print(c_matrix)
        print("Class acc:", class_acc)
        print("Accuracy: %.5f" % acc)
        print("F1-macro:", f1_macro)
        print("F1-micro:", f1_micro)
        print("Precision", precision)
        print("Recall", recall)

        print(actual_labels, '\n', predicted_labels)
        return class_acc, acc, f1_macro, f1_micro, precision, recall
