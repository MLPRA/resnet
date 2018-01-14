import chainer
from chainer import Variable
from chainer.dataset import concat_examples
from chainer.functions import reshape
from chainer.utils import imgproc
from chainer.links.model.vision.resnet import prepare
from chainer.functions.math.sum import sum


class ResNet50Layers(chainer.links.ResNet50Layers):
    def predict(self, images, oversample=True):
        x = concat_examples([prepare(img, size=(256, 256)) for img in images])
        if oversample:
            x = imgproc.oversample(x, crop_dims=(224, 224))
        else:
            x = x[:, :, 16:240, 16:240]
        # Use no_backprop_mode to reduce memory consumption
        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            x = Variable(self.xp.asarray(x))
            layers = ['conv1', 'bn1', 'res2', 'res3', 'res4', 'res5', 'fc6', 'prob']
            y = self(x, layers=layers)
            y = y['prob']
            if oversample:
                n = y.data.shape[0] // 10
                y_shape = y.data.shape[1:]
                y = reshape(y, (n, 10) + y_shape)
                y = sum(y, axis=1) / 10
        return y