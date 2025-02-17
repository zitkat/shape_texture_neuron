import glob
import random
import cv2
import numpy as np
import torch.utils.data as data
import torch

from pathlib import Path

# Stylized VOC
def get_svoc_data(config):
    data_path = Path(config.data_path)
    files = data_path.glob("*")
    data = []
    data_id_list = {}
    for i in range(1, 21):
        data_id_list[i] = {'0': [], '1': [],'2': [],'3': [],'4': [],'5': []}
        # data_id_list[i] = {'1': [],'2': [],'3': [],'4': [], '5': []}
    for i, file in enumerate(files):
        img_file_name = file.name
        cls_label = img_file_name.split('_')[1]
        tex_id = img_file_name[0]
        img_id = img_file_name.split('_')[2] + '_' + img_file_name.split('_')[3]
        data_id_list[int(cls_label)][tex_id].append(file)
        sample = {
            'cls': int(cls_label),
            'texture': int(tex_id),
            'img_id': img_id,
            'file_path': file,
        }
        if not tex_id == '0':
            data.append(sample)


    return data, data_id_list

class StylizedVoc(data.Dataset):
    def __init__(self, config):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        # self.data = get_coco_data()
        self.data, self.data_ids = get_svoc_data(config)
        self.num_textures = 5
        self.n_factors = config.n_factors
        self.num_classes = 20
        self.image_size = config.image_size  # int(config.model.split('_')[-1])
        self.list_possible_shapes = []
        for key in self.data_ids:
            if len(self.data_ids[key]['0']):
                self.list_possible_shapes.append(key)
        self.prng = np.random.RandomState(1)

    def get_random_factor(self, i):
        factor = self.prng.choice(2)
        return factor

    def get_image(self, path):
        # open image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = image.astype(np.float32)
        # resize
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # normalize (imagenet norm used)
        image = self._normalize(image, mean=[0.456, 0.406, 0.485], std=[0.224, 0.225, 0.229])
        # convert to tensor and reshape
        image = np.array(image)
        torch.tensor(image).permute(2,0,1)
        return torch.tensor(image).permute(2,0,1)

    def _normalize(self, image, mean=(0., 0., 0.), std=(1., 1., 1.)):
        if mean[0] < 1:
            image /= 255.0
        image -= mean
        image /= std
        return image

    def __getitem__(self, i):  # shape and texture
        # use text file and open example1 and example2 based on self.data txt file
        data1 = self.data[i]
        path1 = data1['file_path']
        id1 = data1['img_id']
        cls1 = data1['cls']
        texture1 = data1['texture']
        example1 = self.get_image(path1)
        factor = random.randint(0, self.n_factors)
        example = {"factor": factor, "example1": {'image': example1, 'class': cls1}}

        # select random factor (0 is shape, 1 is texture)
        if factor == 0:
            # same shape, different texture
            list_possible_textures = list(range(1, self.num_textures+1))
            # select same image with different texture
            list_possible_textures.remove(texture1)
            new_texture = random.choice(list_possible_textures)
            id2 = str(new_texture) + path1.name[1:]
            path2 = path1.parent / id2
            cls2 = cls1
        else:
            # different shape (class), same texture
            list_possible_shapes = self.list_possible_shapes.copy()
            # select different image with same texture
            list_possible_shapes.remove(cls1)
            new_shape = random.choice(list_possible_shapes)
            choose_new_file_list = self.data_ids[new_shape][str(texture1)]
            cls2 = new_shape
            path2 = random.choice(choose_new_file_list)

        example2 = self.get_image(path2)

        return factor, example1, example2, cls1, cls2

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    class Config():
        n_factors = 5
        data_path = "D:\Datasets\STYLIZED_VOC2012"
        image_size = 513

    dt = StylizedVoc(Config())

    from shutil import copyfile
    for id, data in dt.data_ids.items():
        for d, paths in data.items():
            copyfile(paths[0], Path(r"D:\Datasets\MINI_STYLIZED_VOC2012") / paths[0].name)

