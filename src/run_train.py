from argparse import ArgumentParser
import chainer
import multiprocessing
from chainer.links import ResNet50Layers
from progressbar import ProgressBar
from src.dataset import LabeledImageDataset

def run_train():

    parser = ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--paths', type=str, nargs='+', required=True)

    args = parser.parse_args()

    device = 0 if args.gpu >= 0 else -1

    model = ResNet50Layers(None)

    # load npz stuff here
    # chainer.serializers.load_npz('{}.npz'.format('resnet50'), model)

    model = chainer.links.Classifier(model)

    # gpu
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = LabeledImageDataset(args.paths)

    multiprocessing.set_start_method('spawn')

    dataset_iter = chainer.iterators.MultiprocessIterator(dataset, args.batchsize, repeat=False)

    # what does this mean ?
    chainer.config.enable_backprop = False
    chainer.config.train = False

    sum_accuracy = 0
    count = 0

    progress = ProgressBar(min_value=0, max_value=len(dataset)).start()
    for batch in dataset_iter:

        x, t = chainer.dataset.concat_examples(batch, device=device)
        x = chainer.Variable(x)
        t = chainer.Variable(t)
        loss = model(x, t)
        sum_accuracy += float(model.accuracy.data)
        count += len(t)

        progress.update(count)

    dataset_iter.finalize()