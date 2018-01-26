from argparse import ArgumentParser
from sklearn.manifold import TSNE
import numpy as np
import os

def run_tsne():

    parser = ArgumentParser()
    parser.add_argument('--vectors', type=str, required=True,
                        help='Path to feature vectors file')
    parser.add_argument('--out', type=str, required=True,
                        help='Output folder for 2D feature vector')
    args = parser.parse_args()

    npz_file = np.load(args.vectors)
    feature_vectors = npz_file['arr_0']
    labels = npz_file['arr_1']

    tsne = TSNE()

    feature_vectors_2d = tsne.fit_transform(feature_vectors)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    vectors_output = '{}/vectors_2d.npz'.format(args.out)
    np.savez(vectors_output, feature_vectors_2d, labels)