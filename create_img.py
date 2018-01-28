from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.cm as cm

from src.dataset import LabelHandler

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vectors', type=str, required=True,
                        help='path to 2d features vector file')
    parser.add_argument('--label_names', type=str, required=True,
                        help='Path to label names file')
    parser.add_argument('--out', type=str, required=True,
                        help='Output folder for the image')
    args = parser.parse_args()

    npz_file = np.load(args.vectors)
    feature_vectors = npz_file['arr_0']
    labels = npz_file['arr_1']

    label_handler = LabelHandler(args.label_names)
    cmap = pyplot.cm.get_cmap('hsv', len(label_handler))

    pyplot.figure(figsize=(10, 10))

    for i, (x, y) in enumerate(feature_vectors):
        pyplot.plot(x, y, 'o', color=cmap(labels[i]), markersize=3)

    legend_handles = []
    for label in range(len(label_handler)):
        patch = patches.Patch(color=cmap(label), label=label_handler.get_label_str(label))
        legend_handles.append(patch)
    pyplot.legend(handles=legend_handles)

    file_name_arr = args.vectors.split('/')[-1].split('.')
    file_name = '.'.join(file_name_arr[0:-1])

    pyplot.savefig('{}/{}.png'.format(args.out, file_name))