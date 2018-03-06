import json
from argparse import ArgumentParser
import chainer

from src.evaluator import Evaluator
from src.resnet import ResNet50Layers
from chainer.training import extensions

from src.classifier import Classifier
from src.dataset import LabeledImageDatasetBuilder, LabelHandler


def run_train():

    parser = ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', required=True,
                        help='Root paths of folders that contain images and pascal voc files')
    parser.add_argument('--label_names', type=str, required=True,
                        help='Path to label names file')
    parser.add_argument('--max_images', type=int, default=-1,
                        help='Max images per class')
    parser.add_argument('--training_splitsize', type=float, default=0.9,
                        help='Splitsize of training data')
    parser.add_argument('--batchsize', type=int, default=100,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Numbers of epochs to train')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID, negative value indicates CPU')
    parser.add_argument('--out', default='trainer_output',
                        help='Output directory of trainer')
    parser.add_argument('--val_batchsize', type=int, default=100,
                        help='Validation minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = parser.parse_args()

    # create model
    predictor = ResNet50Layers(None)
    model = Classifier(predictor)

    # use selected gpu by id
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # build datasets from paths
    label_handler = LabelHandler(args.label_names)
    builder = LabeledImageDatasetBuilder(args.paths, label_handler)
    if args.max_images > 0:
        builder.even_dataset(args.max_images)

    train_dataset, val_dataset = builder.get_labeled_image_dataset_split(args.training_splitsize)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize, repeat=False)

    output_dir = '{}/{}_{}_{}_{}'.format(args.out, args.batchsize, args.epoch, args.learning_rate, args.momentum)

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(args.learning_rate, args.momentum)
    optimizer.setup(model)

    # trainer
    updater = chainer.training.updater.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), output_dir)

    trainer.extend(extensions.LogReport())
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))
    evaluator = Evaluator(val_iter, model, label_handler, device=args.gpu)
    trainer.extend(evaluator)
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))

    accuracy_keys = ['validation/main/{}_accuracy'.format(label_name) for label_name in label_handler.label_names]
    trainer.extend(
        extensions.PlotReport(list(set().union(accuracy_keys, ['main/accuracy', 'validation/main/accuracy'])), x_key='epoch', file_name='accuracy.png'))

    trainer.run()

    # save model
    output_file_path = '{0}/resnet.model'.format(output_dir)
    chainer.serializers.save_npz(output_file_path, predictor)

    # save meta information
    images = train_dataset.get_image_numbers(range(len(label_handler)))
    images_per_label = {}
    for label in range(len(label_handler)):
        images_per_label[label_handler.get_label_str(label)] = images[label]

    meta_output = {
        'data/images': len(train_dataset),
        'data/images_per_label': images_per_label,
        'training/epochs': args.epoch,
        'training/momentum': args.momentum,
        'training/learning_rate': args.learning_rate,
        'training/batchsize': args.batchsize,
    }

    with open('{0}/meta.json'.format(output_dir), 'w') as f:
        json.dump(meta_output, f, indent=4)
