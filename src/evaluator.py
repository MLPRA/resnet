import chainer
import copy

import numpy
from chainer import reporter as reporter_module
from chainer.dataset import convert


class Evaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, target, label_handler, converter=convert.concat_examples,
                 device=None):
        super(Evaluator, self).__init__(iterator, target, converter=converter,
                                        device=device)
        self.label_handler = label_handler

    def evaluate(self):
        observation = super().evaluate()

        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
        else:
            iterator = copy.copy(iterator)

        tp = numpy.zeros((1000,), dtype=int)
        fp = numpy.zeros((1000,), dtype=int)

        for batch in iterator:
            imgs = []
            gt_values = []
            for sample in batch:
                imgs.append(sample[0])
                gt_values.append(sample[1])

            gt_values = numpy.asarray(gt_values)

            pred_values = target.predictor.predict(imgs)
            pred_values = numpy.argmax(pred_values.array, axis=1)

            for i, value in enumerate(gt_values):
                if value == int(pred_values[i]):
                    tp[value] += 1
                else:
                    fp[value] += 1

        report = {}
        for label in range(len(self.label_handler)):
            if tp[label] > 0 or fp[label] > 0:
                report[self.label_handler.get_label_str(label) + '_accuracy'] = tp[label] / (tp[label] + fp[label])

        with reporter_module.report_scope(observation):
            reporter_module.report(report, target)

        return observation