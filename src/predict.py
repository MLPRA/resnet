from argparse import ArgumentParser

import chainer

from src.dataset import ImageSegment
from src.resnet import ResNet50Layers


def predict():

    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='Path of image')
    parser.add_argument('--xmin', type=int, default=-1,
                        help='Minuimum x value of slice')
    parser.add_argument('--xmax', type=int, default=-1,
                    help='Maximum x value of slice')
    parser.add_argument('--ymin', type=int, default=-1,
                        help='Minuimum y value of slice')
    parser.add_argument('--ymax', type=int, default=-1,
                        help='Maximum y value of slice')
    parser.add_argument('--model', type=str, required=True,
                        help='Path of model for the resnet')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID, negative value indicates CPU')
    args = parser.parse_args()

    model = ResNet50Layers(pretrained_model=args.model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    image_segment = ImageSegment(args.image, args.xmin, args.ymin, args.xmax, args.ymax)

    image = image_segment()

    prediction = model.predict([image], oversample=True)

    print(prediction)