from argparse import ArgumentParser

import chainer

from src.dataset import ImageSegment, LabelHandler
from src.resnet import ResNet50Layers


def predict():

    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='Path of image')
    parser.add_argument('--xmin', type=int, required=True,
                        help='Minuimum x value of slice')
    parser.add_argument('--xmax', type=int, required=True,
                    help='Maximum x value of slice')
    parser.add_argument('--ymin', type=int, required=True,
                        help='Minuimum y value of slice')
    parser.add_argument('--ymax', type=int, required=True,
                        help='Maximum y value of slice')
    parser.add_argument('--model', type=str, required=True,
                        help='Path of model for the resnet')
    parser.add_argument('--label_names', type=str, required=True,
                        help='Path to label names file')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID, negative value indicates CPU')
    args = parser.parse_args()

    model = ResNet50Layers(pretrained_model=args.model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    label_handler = LabelHandler(args.label_names)

    image_segment = ImageSegment(args.image, args.xmin, args.ymin, args.xmax, args.ymax)

    image = image_segment()

    prediction = model.predict([image], oversample=True)[0].array

    for i in range(len(prediction)):
        if prediction[i] > 0:
            output = '{}: {}%'.format(label_handler.get_label_str(i), prediction[i] * 100)
            print(output)