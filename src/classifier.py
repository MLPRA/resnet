import chainer
import numpy
from chainer import reporter

from src.class_accuracy import class_accuracy


class Classifier(chainer.links.Classifier):
    def __call__(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*args, **kwargs, layers=['fc6', 'prob'])
        self.loss = self.lossfun(self.y['fc6'], t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y['prob'], t)
            reporter.report({'accuracy': self.accuracy}, self)
            accuracy = {'accuracy_' + str(i): a for i, a in enumerate(class_accuracy(self.y['prob'], t)) if numpy.isfinite(a)}
            reporter.report(accuracy, self)
        return self.loss
