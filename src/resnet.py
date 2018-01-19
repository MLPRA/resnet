import chainer
from chainer import Variable
from chainer.dataset import concat_examples
from chainer.links.model.vision.resnet import prepare

class ResNet50Layers(chainer.links.ResNet50Layers):

    def _layer_out(self, images, layer):
        layers = ['conv1', 'bn1', 'res2', 'res3', 'res4', 'res5', 'fc6', 'prob']
        if layer not in layers:
            raise ValueError('Layer {} does not exist.'.format(layer))

        x = concat_examples([prepare(img) for img in images])

        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            x = Variable(self.xp.asarray(x))

            y = self(x, layers=[layer])[layer]
        return y

    def feature_vector(self, images):
        return self._layer_out(images, 'res5')

    def predict(self, images):
        return self._layer_out(images, 'prob')