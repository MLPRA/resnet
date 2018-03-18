import configparser
import json
from argparse import ArgumentParser
import chainer
from chainer.training.extensions import Evaluator

from src.resnet import ResNet50Layers
from chainer.training import extensions

from src.classifier import Classifier
from src.dataset import LabeledImageDatasetBuilder, LabelHandler


def run_train():
    parser = ArgumentParser()
    parser.add_argument('--settings', type=str, required=True,
                        help='Path to the training settings ini file')

    settings = configparser.ConfigParser()
    settings.read(parser.parse_args().settings)

    # create model
    predictor = ResNet50Layers(None)
    model = Classifier(predictor)

    # use selected gpu by id
    gpu = settings.getint('hardware', 'gpu')
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    label_handler, train_dataset, val_dataset = _create_datasets(settings['input_data'])

    train_iter = chainer.iterators.SerialIterator(train_dataset, settings.getint('trainer', 'batchsize'))
    val_iter = chainer.iterators.SerialIterator(val_dataset, settings.getint('trainer', 'batchsize'), repeat=False)

    output_dir = '{}/training_{}_{}'.format(settings.get('output_data', 'path'), settings.get('trainer', 'epochs'), settings.get('optimizer', 'optimizer'))

    # optimizer
    optimizer = _create_optimizer(settings['optimizer'])
    optimizer.setup(model)

    # trainer
    updater = chainer.training.updater.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = chainer.training.Trainer(updater, (settings.getint('trainer', 'epochs'), 'epoch'), output_dir)

    trainer.extend(extensions.LogReport())
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))
    evaluator = Evaluator(val_iter, model, device=gpu)
    trainer.extend(evaluator)
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    trainer.run()

    # save model
    output_file_path = '{0}/resnet.model'.format(output_dir)
    chainer.serializers.save_npz(output_file_path, predictor)

    meta_output = {
        'trainer': settings._sections['trainer'],
        'optimizer': settings._sections['optimizer'],
        'train_data': train_dataset.get_meta_info(label_handler),
        'validation_data': val_dataset.get_meta_info(label_handler),
    }

    with open('{0}/meta.json'.format(output_dir), 'w') as f:
        json.dump(meta_output, f, indent=4)


def _create_datasets(input_data):
    paths = input_data.get('paths').split()
    label_names = input_data.get('label_names')
    max_images = input_data.getint('max_images')
    split = input_data.getfloat('split')

    label_handler = LabelHandler(label_names)
    builder = LabeledImageDatasetBuilder(paths, label_handler)
    if max_images > 0:
        builder.even_dataset(max_images)

    train_dataset, val_dataset = builder.get_labeled_image_dataset_split(split)

    return label_handler, train_dataset, val_dataset


def _create_optimizer(optimizer_data):
    optimizer_class = getattr(chainer.optimizers, optimizer_data.get('optimizer'))

    optimizer_args = {}
    for key in optimizer_data:
        if key == 'optimizer':
            continue
        optimizer_args[key] = optimizer_data.getfloat(key)

    optimizer = optimizer_class(**optimizer_args)

    return optimizer
