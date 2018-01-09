from argparse import ArgumentParser
import chainer
import multiprocessing
from chainer.links import ResNet50Layers
from progressbar import ProgressBar
from src.dataset import LabeledImageDatasetBuilder

def run_train():

    parser = ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', required=True,
                        help='Root paths of folders that contain images and pascal voc files')
    parser.add_argument('--training_splitsize', type=float, default=0.9,
                        help='Splitsize of training data')
    parser.add_argument('--batchsize', type=int, default=20,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Numbers of epochs to train')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID, negative value indicates CPU')
    parser.add_argument('--out', default='output',
                        help='Output directory')
    parser.add_argument('--val_batchsize', type=int, default=250,
                        help='Validation minibatch size')
    args = parser.parse_args()

    # create model
    model = ResNet50Layers(None)

    # TODO: initmodel

    # use selected gpu by id
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # TODO: mean

    # build datasets from paths
    builder = LabeledImageDatasetBuilder(args.paths)
    train_dataset, val_dataset = builder.get_labeled_image_dataset_split(args.training_splitsize)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize, repeat=False)
    val_iter = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize, repeat=False)

    # optimizer
    learning_rate = 0.01
    momentum = 0.9
    optimizer = chainer.optimizers.MomentumSGD(learning_rate, momentum)
    optimizer.setup(model)

    # trainer
    updater = chainer.training.updater.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    # TODO: extensions

    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    trainer.run()