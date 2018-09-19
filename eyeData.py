from __future__ import print_function, division, absolute_import, unicode_literals

from tf_unet.image_util import BaseDataProvider
import glob
import numpy as np
from PIL import Image
import tensorflow as tf


class EyeDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, search_path, a_min=None, a_max=None, shuffle_data=True, n_class=2):
        super(EyeDataProvider, self).__init__(a_min, a_max)
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class

        self.data_files = self._find_data_files(search_path)
        self.data_num = len(self.data_files)


        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        assert self.data_num > 0, "No training files"
        print("Number of files used: %s" % self.data_num)

        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return all_files

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def reset_data(self):
        self.file_idx = -1
        if self.shuffle_data:
            np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = self._from_image_to_label_name(image_name)

        img = self._load_file(image_name, np.float32)
        new_shape = (int(img.shape[0] / 4), int(img.shape[1] / 4))
        img = np.resize(img, new_shape)
        img = tf.random_crop(img, [112, 112, self.channels])
        label = self._load_file(label_name, np.uint8)
        if self.n_class > 2:
            label = self._convert_label_to_onehot(label)
        else:
            label = (label == 255)


        return img, label

    def _from_image_to_label_name(self, image_name):
        # image name: /Users/miao/Downloads/eyedata/Edema_trainingset/original_images/PC008_MacularCube512x128_9-12-2013_10-50-5_OD_sn9624_cube_z.img/28.bmp
        # label name: /Users/miao/Downloads/eyedata/Edema_trainingset/label_images/PC008_MacularCube512x128_9-12-2013_10-50-5_OD_sn9624_cube_z_labelMark/28.bmp
        label_name = image_name.replace("original_", "label_").replace("_z.img", "_z_labelMark")
        return label_name

    def _convert_label_to_onehot(self, label):
        class_gray = [0, 128, 191, 255]
        label_shape = label.shape
        new_label = np.zeros(shape=(label_shape[0], label_shape[1], len(class_gray)), dtype=np.bool)
        for i in range(len(class_gray)):
            new_label[:, :, i] = (label == class_gray[i])
        return new_label






if __name__=="__main__":
    print(glob.glob("/Users/miao/Downloads/eyedata/Edema_?????*set"))
