import torch.utils.data as data
import numpy as np

from PIL import Image
from collections import Counter
import os
import os.path

from torchvision import transforms
import torch

from timm.data import create_transform
from ..utils import logging
logger = logging.get_logger("visual_prompt")



_FOLDER_MAP = {
    'caltech101': 'caltech101',
    'cifar(num_classes=100)': 'cifar',
    'dtd': 'dtd',
    'oxford_flowers102': 'oxford_flowers102',
    'oxford_iiit_pet': 'oxford_iiit_pet',
    'patch_camelyon': 'patch_camelyon',
    'sun397': 'sun397',
    'svhn': 'svhn',
    'resisc45': 'resisc45',
    'eurosat': 'eurosat',
    'dmlab': 'dmlab',
    'kitti(task="closest_vehicle_distance")': 'kitti',
    'smallnorb(predicted_attribute="label_azimuth")': 'smallnorb_azi',
    'smallnorb(predicted_attribute="label_elevation")': 'smallnorb_ele',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': 'dsprites_loc',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': 'dsprites_ori',
    'clevr(task="closest_object_distance")': 'clevr_dist',
    'clevr(task="count_all")': 'clevr_count',
    'diabetic_retinopathy(config="btgraham-300")': 'diabetic_retinopathy',
}



def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_data(basedir, name, logger=None, evaluate=True, train_aug=False, batch_size=64):
    root = os.path.join(basedir, name)
    
    if train_aug:
        aug_transform = create_transform(
                input_size=(224, 224),
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                re_prob=0.0,
                re_mode='pixel',
                re_count=1,
                interpolation='bicubic',
            )
        aug_transform.transforms[0] = transforms.Resize((224, 224), interpolation=3)
    else:
        aug_transform = None

    # print(aug_transform)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform = aug_transform if aug_transform else transform
    # train_transform = transform

    if logger is not None:
        logger.info(f'Data transform, train:\n{train_transform}')
        logger.info(f'Data transform, test:\n{transform}')

    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/train800val200.txt",
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/test.txt",
                transform=transform),
            batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/train800.txt",
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/val200.txt",
                transform=transform),
            batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)
        
    return train_loader, val_loader


class VTabData(data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
            "trainval"
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))
        
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME

        # if split == "train" or split == "trainval":
        #     self.transform = create_transform(
        #         input_size=(224, 224),
        #         is_training=True,
        #         color_jitter=0.4,
        #         auto_augment='rand-m9-mstd0.5-inc1',
        #         re_prob=0.0,
        #         re_mode='pixel',
        #         re_count=1,
        #         interpolation='bicubic',
        #     )
        #     self.transform.transforms[0] = transforms.Resize((224, 224), interpolation=3)
        # else:
        self.transform =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        vtab_dataname = cfg.DATA.NAME.split("vtab-")[-1]
        root = os.path.join(cfg.DATA.DATAPATH, _FOLDER_MAP[vtab_dataname])

        self.root = root

        if split == "train":
            flist = os.path.join(root, "train800.txt")
        elif split == "val":
            flist = os.path.join(root, "val200.txt")
        elif split == "test":
            flist = os.path.join(root, "test.txt")
        else:
            flist = os.path.join(root, "train800val200.txt")

        self.imlist = default_flist_reader(flist)
        self.loader = default_loader


        self._image_tensor_list = []
        self._targets = []
        for impath, target in self.imlist:
            img = self.loader(os.path.join(self.root, impath))
            self._image_tensor_list.append(img)
            self._targets.append(target)

        self._class_ids = sorted(list(set(self._targets)))

        logger.info("Number of images: {}".format(len(self._image_tensor_list)))
        logger.info("Number of classes: {} / {}".format(
            len(self._class_ids), self.get_class_num()))


    def get_info(self):
        num_imgs = len(self._image_tensor_list)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        label = self._targets[index]
        im = self._image_tensor_list[index]
        im = self.transform(im)

        sample = {
            "image": im,
            "label": label,
            # "id": index
        }
        return sample

    def __len__(self):
        return len(self.imlist)