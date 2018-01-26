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
    dataset = LabeledImageDatasetBuilder(args.paths, label_handler).get_labeled_image_dataset()

    # feature_vector_label_pairs = []
    # image, label = dataset[0]
    # feature_vector = model.feature_vector([image])[0].array
    # feature_vector_label_pairs.append((feature_vector, int(label)))

    feature_vector_label_pairs = [(model.feature_vector([image])[0].array, int(label)) for image, label in dataset]

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    output_file_path = '{}/feature_vectors.npz'.format(args.out)
    np.savez(output_file_path, feature_vector_label_pairs)

    print('finished')