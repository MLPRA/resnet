import numpy


def class_accuracy(y, t):
    numpy.seterr(divide='ignore', invalid='ignore')

    pred = y.array.argmax(axis=1).reshape(t.shape)
    correct_pred = pred == t

    tp = numpy.zeros((1000,), dtype=int)
    total = numpy.zeros((1000,), dtype=int)

    for label, correct in zip(t, correct_pred):
        label = int(label)
        total[label] += 1
        if correct:
            tp[label] += 1

    return tp/total
