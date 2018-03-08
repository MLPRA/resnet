import configparser
from argparse import ArgumentParser

import chainer
import os
from matplotlib import pyplot, patches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.dataset import LabelHandler, LabeledImageDatasetBuilder
from src.resnet import ResNet50Layers


def run_visualization():
    parser = ArgumentParser()
    parser.add_argument('--settings', type=str, required=True,
                        help='Path to the visualization settings ini file')

    settings = configparser.ConfigParser()
    settings.read(parser.parse_args().settings)

    model = ResNet50Layers(pretrained_model=settings.get('input_data', 'model'))

    gpu = settings.getint('hardware', 'gpu')
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    label_handler = LabelHandler(settings.get('input_data', 'label_names'))
    builder = LabeledImageDatasetBuilder(settings.get('input_data', 'paths').split(), label_handler)
    builder.even_dataset(settings.getint('input_data', 'max_images'))
    dataset = builder.get_labeled_image_dataset()
    dataset_iter = chainer.iterators.SerialIterator(dataset, 100, repeat=False, shuffle=False)

    # feature vectors
    feature_vectors = []
    labels = []

    with chainer.no_backprop_mode():
        for batch in dataset_iter:
            imgs, batch_labels = zip(*batch)

            feature_vectors.extend(model.feature_vector(imgs))
            labels.extend(batch_labels)

    feature_vectors = [x.array for x in feature_vectors]
    labels = [int(x) for x in labels]

    # dimension reduction
    pca = PCA(settings.getint('visualization', 'pca_dimensions'))
    feature_vectors_pca = pca.fit_transform(feature_vectors)

    tsne = TSNE(n_components=settings.getint('visualization', 'tsne_dimensions'), verbose=True)
    feature_vectors_tsne = tsne.fit_transform(feature_vectors_pca)

    # visualization
    cmap = pyplot.cm.get_cmap('rainbow', len(label_handler))
    pyplot.figure(figsize=(10, 10))

    tsne_dimensions = settings.getint('visualization', 'tsne_dimensions')
    if tsne_dimensions == 2:
        for feature_vector, label in zip(feature_vectors_tsne, labels):
            pyplot.plot(feature_vector[0], feature_vector[1], 'o', color=cmap(label), markersize=3)

        legend_handles = []
        for label_name, label_int in label_handler.label_names_dict.items():
            legend_handles.append(patches.Patch(color=cmap(label_int), label=label_name))
        pyplot.legend(handles=legend_handles)

        output_dir = settings.get('output_data', 'path')
        output_filename = 'pca{}_tsne{}.png'.format(settings.get('visualization', 'pca_dimensions'), settings.get('visualization', 'tsne_dimensions'))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pyplot.savefig('{}/{}'.format(output_dir, output_filename))

    elif tsne_dimensions == 3:
        # TODO
        pass
