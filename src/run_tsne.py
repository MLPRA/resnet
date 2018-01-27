from argparse import ArgumentParser
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os

def run_tsne():

    parser = ArgumentParser()
    parser.add_argument('--vectors', type=str, required=True,
                        help='Path to feature vectors file')
    parser.add_argument('--pca_components', type=int, required=True,
                        help='Number of components for the PCA before the TSNA')
    parser.add_argument('--out', type=str, required=True,
                        help='Output folder for 2D feature vector')
    args = parser.parse_args()

    npz_file = np.load(args.vectors)
    feature_vectors = npz_file['arr_0']
    labels = npz_file['arr_1']

    pca = PCA(args.pca_components)
    feature_vectors_principal_components = pca.fit_transform(feature_vectors)

    tsne = TSNE()
    feature_vectors_2d = tsne.fit_transform(feature_vectors_principal_components)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    file_name = args.vectors.split('/')[-1].split('.')[0]

    vectors_output = '{}/{}_{}_2d.npz'.format(args.out, file_name, args.pca_components)
    np.savez(vectors_output, feature_vectors_2d, labels)