from argparse import ArgumentParser
import numpy as np
import chainer
import os

from src.dataset import LabelHandler, LabeledImageDatasetBuilder
from src.resnet import ResNet50Layers


def run_feature_vectors():

    parser = ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', required=True,
                        help='Root paths of folders that contain images and pascal voc files')
    parser.add_argument('--out', default='feature_vectors',
                        help='Output directory of feature vectors')
    parser.add_argument('--label_names', type=str, required=True,
                        help='Path to label names file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path of model for the resnet')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID, negative value indicates CPU')
    args = parser.parse_args()

    model = ResNet50Layers(pretrained_model=args.model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    label_handler = LabelHandler(args.label_names)
    builder = LabeledImageDatasetBuilder(args.paths, label_handler)
    builder.even_dataset(500)
    dataset = builder.get_labeled_image_dataset()

    feature_vectors = []
    labels = []
    for i, (image, label) in enumerate(dataset):
        print(i)
        feature_vectors.append(model.feature_vector([image])[0].array)
        labels.append(label)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    vectors_output = '{}/vectors.npz'.format(args.out)
    np.savez(vectors_output, feature_vectors, labels)
