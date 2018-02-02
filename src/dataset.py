import json

import cv2
import os
import numpy as np
import random
import xml.etree.ElementTree as ElementTree

from PIL import Image


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
        image_segement, label = self.labeled_images[index]
        label = np.array(label, dtype=np.int32)

        return image_segement(), label

    def get_image_numbers(self, arr):
        numbers = {}
        for label in arr:
            numbers[label] = 0

        for _, label in self.labeled_images:
            if label in numbers:
                numbers[label] += 1

        out = []
        for label in arr:
            out.append(numbers[label])

        return out


class ImageSegment:

    IMG_SIZE = 224

    def __init__(self, path, xmin, ymin, xmax, ymax):
        self.path = path
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __call__(self):
        if not os.path.exists(self.path):
            raise IOError('File does not exist: %s' % self.path)

        pil_image = Image.open(self.path)
        pil_image = pil_image.convert('RGB')
        pil_image = pil_image.crop((self.xmin, self.ymin, self.xmax, self.ymax))
        pil_image = pil_image.resize((self.IMG_SIZE, self.IMG_SIZE))

        image = np.asarray(pil_image, dtype=np.float32)
        image = image[:, :, ::-1]

        image -= np.array([103.063, 115.903, 123.152], dtype=np.float32)
        image = image.transpose((2, 0, 1))
        return image


class LabeledImageDatasetBuilder:

    def __init__(self, dir_paths, label_handler, type='xml'):
        type = type.lower()
        if type not in ['xml', 'json']:
            raise ValueError('Type needs to be xml or json. Given: {}'.format(type))

        self.images = []

        # crawl images
        jpg_paths = {}
        xml_paths = {}
        json_paths = {}

        for dir_path in dir_paths:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension.lower() == '.jpg':
                        jpg_paths[file_name] = '{}/{}'.format(root, file)
                    elif file_extension.lower() == '.xml' and type == 'xml':
                        xml_paths[file_name] = '{}/{}'.format(root, file)
                    elif file_extension.lower() == '.json' and type == 'json':
                        json_paths[file_name] = '{}/{}'.format(root, file)

        # map images to labels and bounding boxes
        if type == 'xml':
            self._create_images_xml(jpg_paths, xml_paths, label_handler)
        elif type == 'json':
            self._create_images_json(jpg_paths, json_paths, label_handler)

        random.shuffle(self.images)


    def _create_images_xml(self, jpg_paths, xml_paths, label_handler):
        for key in xml_paths:
            if key in jpg_paths:
                xml = ElementTree.parse(xml_paths[key])
                for object in xml.findall('object'):
                    object_name = object.find('name').text

                    if not label_handler.is_label_str(object_name):
                        continue

                    object_label = label_handler.get_label_int(object_name)

                    object_xmin = int(object.find('bndbox/xmin').text)
                    object_ymin = int(object.find('bndbox/ymin').text)
                    object_xmax = int(object.find('bndbox/xmax').text)
                    object_ymax = int(object.find('bndbox/ymax').text)

                    image_segment = ImageSegment(jpg_paths[key], object_xmin, object_ymin, object_xmax,
                                                 object_ymax)
                    self.images.append((image_segment, object_label))

    def _create_images_json(self, jpg_paths, json_paths, label_handler):
        for key in json_paths:
            if key in jpg_paths:
                with open(json_paths[key]) as f:
                    data = json.load(f)
                object_name = data['label']

                if not label_handler.is_label_str(object_name):
                    continue

                object_label = label_handler.get_label_int(object_name)

                with Image.open(jpg_paths[key]) as img:
                    width, height = img.size

                object_xmin = int(data['boundingBox']['x'] * width)
                object_ymin = int(data['boundingBox']['y'] * height)
                object_xmax = int(data['boundingBox']['x'] + data['boundingBox']['width'] * width)
                object_ymax = int(data['boundingBox']['y'] + data['boundingBox']['height'] * height)

                image_segment = ImageSegment(jpg_paths[key], object_xmin, object_ymin, object_xmax, object_ymax)
                self.images.append((image_segment, object_label))

    def even_dataset(self, max_per_label=None):
        if max_per_label == None:
            label_counter = {}
            for _, label in self.images:
                if label in label_counter:
                    label_counter[label] += 1
                else:
                    label_counter[label] = 1
            max_per_label = min(label_counter.values())

        new_images = []
        label_counter = {}
        for image, label in self.images:
            if label in label_counter:
                if label_counter[label] == max_per_label:
                    continue
                label_counter[label] += 1
            else:
                label_counter[label] = 1
            new_images.append((image, label))

        self.images = new_images

    def eliminate_class(self, eliminate_label):
        new_images = []
        for image, label in self.images:
            if label != eliminate_label:
                new_images.append((image, label))
        self.images = new_images

    def get_labeled_image_dataset(self):
        return LabeledImageDataset(self.images)

    def get_labeled_image_dataset_split(self, splitsize):
        splitnumber = int(round(len(self.images) * splitsize))

        dataset1 = LabeledImageDataset(self.images[:splitnumber])
        dataset2 = LabeledImageDataset(self.images[splitnumber:])

        return dataset1, dataset2


class LabelHandler():

    def __init__(self, label_names):
        with open(label_names, 'r') as f:
            self.label_names = f.read().splitlines()

        self.label_names_dict = {}
        for i in range(len(self.label_names)):
            self.label_names_dict[self.label_names[i]] = i

    def get_label_int(self, label_str):
        return self.label_names_dict[label_str]

    def get_label_str(self, label_int):
        return self.label_names[label_int]

    def is_label_int(self, label_int):
        if label_int in range(len(self.label_names)):
            return True
        return False

    def is_label_str(self, label_str):
        if label_str in self.label_names_dict:
            return True
        return False

    def __len__(self):
        return len(self.label_names)