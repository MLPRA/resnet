import cv2
import os
import numpy as np
import random
import xml.etree.ElementTree as ElementTree


class LabeledImageDataset:

    def __init__(self, labeled_images):
        self.labeled_images = labeled_images

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def get_example(self, index):
        return self.labeled_images[index].img_label_pair()


class LabeledImage:

    IMG_SIZE = 300

    def __init__(self, path, label, xmin, ymin, xmax, ymax):
        self.path = path
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def img_label_pair(self):
        if not os.path.exists(self.path):
            raise IOError('File does not exist: %s' % self.path)

        # Read image and crop area
        image = cv2.imread(self.path)
        image = image[self.ymin:self.ymax, self.xmin:self.xmax, :]

        height, width = image.shape[:2]
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))

        image = image.transpose(2, 0, 1).astype(np.float32)
        image *= (1.0 / 255.0)

        label = np.array(self.label, dtype=np.int32)

        return image, label


class LabeledImageDatasetBuilder:

    def __init__(self, dir_paths):
        self.labeled_images = []

        jpg_paths = {}
        xml_paths = {}

        for dir_path in dir_paths:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension.lower() == '.jpg':
                        jpg_paths[file_name] = root + '/' + file
                    elif file_extension.lower() == '.xml':
                        xml_paths[file_name] = root + '/' + file

        for key in xml_paths:
            if key in jpg_paths:
                xml = ElementTree.parse(xml_paths[key])
                for object in xml.findall('object'):
                    object_name = object.find('name').text
                    object_label = abs(hash(object_name))

                    object_xmin = int(object.find('bndbox/xmin').text)
                    object_ymin = int(object.find('bndbox/ymin').text)
                    object_xmax = int(object.find('bndbox/xmax').text)
                    object_ymax = int(object.find('bndbox/ymax').text)

                    labeled_image = LabeledImage(jpg_paths[key], object_label, object_xmin, object_ymin, object_xmax,
                                                 object_ymax)
                    self.labeled_images.append(labeled_image)

        random.shuffle(self.labeled_images)

    def get_labeled_image_dataset(self):
        return LabeledImageDataset(self.labeled_images)

    def get_labeled_image_dataset_split(self, splitsize):
        splitnumber = int(round(len(self.labeled_images) * splitsize))

        dataset1 = LabeledImageDataset(self.labeled_images[:splitnumber])
        dataset2 = LabeledImageDataset(self.labeled_images[splitnumber:])

        return dataset1, dataset2
