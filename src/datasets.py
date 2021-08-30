import os

import soft_renderer.functional as srf
import torch, random
import numpy as np
import tqdm
from haven import haven_utils as hu
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}

CLASS_IDS = sorted(list(class_ids_map.keys()))
class ShapeNet(object):
    def __init__(self, directory=None, split=None, exp_dict=None):
        self.class_ids = CLASS_IDS
        n_classes = exp_dict.get('n_classes')
        if n_classes:
            self.class_ids = CLASS_IDS[:n_classes]

        classes = exp_dict.get('classes')
        if classes:
            classes_map = {key: value for (value, key) in class_ids_map.items()}
            self.class_ids = sorted([classes_map[k] for k in classes])
            
        self.split = split
        self.elevation = 30.
        self.distance = 2.732
        self.exp_dict = exp_dict
        self.class_ids_map = class_ids_map
        

        self.images = []
        self.voxels = []
        self.labels = []
        self.class_ids_pair = list(zip(self.class_ids, [self.class_ids_map[i] for i in self.class_ids]))

        self.num_data = {}
        self.pos = {}
        count = 0
        # ind2class = {key: value for (value, key) in enumerate(self.class_ids)}
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description(f'Loading {split} Dataset')
        n_train_objects = exp_dict.get('n_train_objects')
        n_ratio_val = exp_dict.get('n_val_ratio')
        # assert n_ratio_val is not None
        if n_train_objects is None and split == 'unlabeled':
            return 
            
        if split in ['train', 'unlabeled']:
            set_name = 'train'
        elif split in ['val', 'test']:
            set_name = 'val'
            if n_ratio_val is  None:
                set_name = split
        for ci, class_id in enumerate(loop):
            i = list(np.load(os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1]
            v = list(np.load(os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1]
            
            
            # train get only first n
            if split == 'train' and n_train_objects is not None:
                n = n_train_objects

                i = i[:n]
                v = v[:n]

            # unlabeled get only first n
            if split == 'unlabeled' and n_train_objects is not None:
                n = n_train_objects

                i = i[n:]
                v = v[n:]

            elif split == 'val' and n_ratio_val is not None:
                n = int(i.shape[0]*n_ratio_val)

                i = i[:n]
                v = v[:n]
                
            elif split == 'test' and n_ratio_val is not None:
                n = int(i.shape[0]*n_ratio_val)

                i = i[n:]
                v = v[n:]

            self.images += [i]
            self.voxels += [v]
            self.labels += [torch.ones(i.shape[0]) * ci]
        self.images = np.concatenate(self.images, axis=0)  
        self.images = torch.from_numpy(self.images.astype('float32') / 255.)

        self.voxels = np.concatenate(self.voxels, axis=0)  
        self.voxels = torch.from_numpy(self.voxels.astype('float32'))

        self.labels = torch.cat(self.labels, dim=0)
        # positible view points
        distances = torch.ones(24).float() * self.distance
        elevations = torch.ones(24).float() * self.elevation
        self.possible_viewpoints = srf.get_points_from_angles(distances, elevations, -torch.arange(24) * 15)

        print(f'{split} samples: {len(self)}')

    def __len__(self):
        if isinstance(self.images, list):
            return len(self.images)
        return self.images.shape[0]

    def __getitem__(self, idx, vp_idx=None, vp_idx_b=None):
        # image A
        images_a, viewpoints_a, viewpoint_id_a = self.get_random_viewpoint(idx, vp_idx)

        # image B
        images_b, viewpoints_b, viewpoint_id_b = self.get_random_viewpoint(idx, vp_idx_b)

        return {'images_a':images_a, 
                'viewpoints_a': viewpoints_a, 
                'object_id_a':idx,
                'viewpoint_id_a':viewpoint_id_a,

                'images_b':images_b, 
                'viewpoints_b': viewpoints_b,
                'object_id_b':idx,
                'viewpoint_id_b':viewpoint_id_b}

    def insert_images(self, images):
        self.images = torch.cat([self.images, images], dim=0)

    def pop_indices(self, ind_list):
        selected_images = self.images[ind_list]
        keep_idx = np.delete(np.arange(self.images.shape[0]), ind_list)
        self.images = self.images[keep_idx]
        # return list(np.delete(arr, id_to_del))
        return selected_images

    def get_random_viewpoint(self, idx, vp_idx=None):
        if vp_idx is None:
            viewpoint_id = np.random.randint(0, 24)
        else:
            viewpoint_id = vp_idx
        # get image and viewpoint
        images = self.images[idx][viewpoint_id]
        
        # get viewpoint
        viewpoints = srf.get_points_from_angles(self.distance, self.elevation, -viewpoint_id * 15)

        return images, torch.as_tensor(viewpoints), viewpoint_id

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        assert self.images.shape[0] == self.voxels.shape[0]

        ci = self.class_ids.index(class_id)
        ind_ci = self.labels == ci
        im_cls = self.images[ind_ci]
        vx_cls = self.voxels[ind_ci]

        data_ids = np.arange(im_cls.shape[0])
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = srf.get_points_from_angles(distances, elevations, -torch.from_numpy(viewpoint_ids).float() * 15)

        shape = im_cls.shape[-3:]
        images = im_cls.view(-1, *shape)

        shape = vx_cls.shape[-3:]
        voxels = vx_cls.view(-1, *shape)

        for i in range((data_ids.size - 1) // batch_size + 1):
            im = images[data_ids[i * batch_size:(i + 1) * batch_size]]
            vx = voxels[data_ids[i * batch_size:(i + 1) * batch_size] // 24]
            yield im, vx



class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
