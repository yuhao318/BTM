import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from .sampler import ClassAwareSampler


class LT_Dataset(Dataset):
    num_classes = 8142

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 
    


class LT_Dataset_Eval(Dataset):
    num_classes = 8142

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 



class BL_Dataset(Dataset):
    num_classes = 8142

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class iNa2018(object):
    def __init__(self, distributed, root="", batch_size=60, num_works=40):
        
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        train_txt = "./datasets/data_txt/iNaturalist18_train.txt"
        eval_txt = "./datasets/data_txt/iNaturalist18_val.txt"

        bl_train_10_0_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_0.txt"
        bl_train_10_1_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_1.txt"
        bl_train_10_2_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_2.txt"
        bl_train_10_3_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_3.txt"
        bl_train_10_4_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_4.txt"
        bl_train_10_5_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_5.txt"
        bl_train_10_6_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_6.txt"
        bl_train_10_7_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_7.txt"
        bl_train_10_8_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_8.txt"
        bl_train_10_9_txt = "./datasets/data_txt/iNaturalist18_BL_train_10_9.txt"

        train_dataset = LT_Dataset(root, train_txt, transform=transform_train)
        eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        
        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)


        bl_train_10_0_dataset = BL_Dataset(root, bl_train_10_0_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_0_dataset = LT_Dataset_Eval(root, bl_train_10_0_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_1_dataset = BL_Dataset(root, bl_train_10_1_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_1_dataset = LT_Dataset_Eval(root, bl_train_10_1_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_2_dataset = BL_Dataset(root, bl_train_10_2_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_2_dataset = LT_Dataset_Eval(root, bl_train_10_2_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_3_dataset = BL_Dataset(root, bl_train_10_3_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_3_dataset = LT_Dataset_Eval(root, bl_train_10_3_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_4_dataset = BL_Dataset(root, bl_train_10_4_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_4_dataset = LT_Dataset_Eval(root, bl_train_10_4_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_5_dataset = BL_Dataset(root, bl_train_10_5_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_5_dataset = LT_Dataset_Eval(root, bl_train_10_5_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_6_dataset = BL_Dataset(root, bl_train_10_6_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_6_dataset = LT_Dataset_Eval(root, bl_train_10_6_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_7_dataset = BL_Dataset(root, bl_train_10_7_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_7_dataset = LT_Dataset_Eval(root, bl_train_10_7_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_8_dataset = BL_Dataset(root, bl_train_10_8_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_8_dataset = LT_Dataset_Eval(root, bl_train_10_8_txt, transform=transform_test, class_map=train_dataset.class_map)

        bl_train_10_9_dataset = BL_Dataset(root, bl_train_10_9_txt, transform=transform_train, class_map=train_dataset.class_map)
        eval_bl_train_10_9_dataset = LT_Dataset_Eval(root, bl_train_10_9_txt, transform=transform_test, class_map=train_dataset.class_map)

        self.bl_train_10_0_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_0_dataset) if distributed else None
        self.bl_train_10_0_instance = torch.utils.data.DataLoader(
            bl_train_10_0_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_0_dist_sampler)

        bl_train_10_0_balance_sampler = ClassAwareSampler(bl_train_10_0_dataset)
        self.bl_train_10_0_balance = torch.utils.data.DataLoader(
            bl_train_10_0_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_0_balance_sampler)

        self.bl_train_10_1_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_1_dataset) if distributed else None
        self.bl_train_10_1_instance = torch.utils.data.DataLoader(
            bl_train_10_1_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_1_dist_sampler)

        bl_train_10_1_balance_sampler = ClassAwareSampler(bl_train_10_1_dataset)
        self.bl_train_10_1_balance = torch.utils.data.DataLoader(
            bl_train_10_1_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_1_balance_sampler)

        self.bl_train_10_2_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_2_dataset) if distributed else None
        self.bl_train_10_2_instance = torch.utils.data.DataLoader(
            bl_train_10_2_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_2_dist_sampler)

        bl_train_10_2_balance_sampler = ClassAwareSampler(bl_train_10_2_dataset)
        self.bl_train_10_2_balance = torch.utils.data.DataLoader(
            bl_train_10_2_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_2_balance_sampler)

        self.bl_train_10_3_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_3_dataset) if distributed else None
        self.bl_train_10_3_instance = torch.utils.data.DataLoader(
            bl_train_10_3_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_3_dist_sampler)

        bl_train_10_3_balance_sampler = ClassAwareSampler(bl_train_10_3_dataset)
        self.bl_train_10_3_balance = torch.utils.data.DataLoader(
            bl_train_10_3_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_3_balance_sampler)

        self.bl_train_10_4_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_4_dataset) if distributed else None
        self.bl_train_10_4_instance = torch.utils.data.DataLoader(
            bl_train_10_4_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_4_dist_sampler)

        bl_train_10_4_balance_sampler = ClassAwareSampler(bl_train_10_4_dataset)
        self.bl_train_10_4_balance = torch.utils.data.DataLoader(
            bl_train_10_4_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_4_balance_sampler)

        self.bl_train_10_5_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_5_dataset) if distributed else None
        self.bl_train_10_5_instance = torch.utils.data.DataLoader(
            bl_train_10_5_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_5_dist_sampler)

        bl_train_10_5_balance_sampler = ClassAwareSampler(bl_train_10_5_dataset)
        self.bl_train_10_5_balance = torch.utils.data.DataLoader(
            bl_train_10_5_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_5_balance_sampler)

        self.bl_train_10_6_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_6_dataset) if distributed else None
        self.bl_train_10_6_instance = torch.utils.data.DataLoader(
            bl_train_10_6_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_6_dist_sampler)

        bl_train_10_6_balance_sampler = ClassAwareSampler(bl_train_10_6_dataset)
        self.bl_train_10_6_balance = torch.utils.data.DataLoader(
            bl_train_10_6_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_6_balance_sampler)

        self.bl_train_10_7_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_7_dataset) if distributed else None
        self.bl_train_10_7_instance = torch.utils.data.DataLoader(
            bl_train_10_7_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_7_dist_sampler)

        bl_train_10_7_balance_sampler = ClassAwareSampler(bl_train_10_7_dataset)
        self.bl_train_10_7_balance = torch.utils.data.DataLoader(
            bl_train_10_7_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_7_balance_sampler)

        self.bl_train_10_8_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_8_dataset) if distributed else None
        self.bl_train_10_8_instance = torch.utils.data.DataLoader(
            bl_train_10_8_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_8_dist_sampler)

        bl_train_10_8_balance_sampler = ClassAwareSampler(bl_train_10_8_dataset)
        self.bl_train_10_8_balance = torch.utils.data.DataLoader(
            bl_train_10_8_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_8_balance_sampler)

        self.bl_train_10_9_dist_sampler = torch.utils.data.distributed.DistributedSampler(bl_train_10_9_dataset) if distributed else None
        self.bl_train_10_9_instance = torch.utils.data.DataLoader(
            bl_train_10_9_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.bl_train_10_9_dist_sampler)

        bl_train_10_9_balance_sampler = ClassAwareSampler(bl_train_10_9_dataset)
        self.bl_train_10_9_balance = torch.utils.data.DataLoader(
            bl_train_10_9_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=bl_train_10_9_balance_sampler)

