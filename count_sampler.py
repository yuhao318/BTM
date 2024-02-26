import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import random

def count_instance(txt, num_classes):
    targets = []
    with open(txt) as f:
        for line in f:
            targets.append(int(line.split()[1]))

    cls_num_list_old = [np.sum(np.array(targets) == i) for i in range(num_classes)]
    print(cls_num_list_old)


    sorted_classes = np.argsort(-np.array(cls_num_list_old))
    class_map = [0 for i in range(num_classes)]
    for i in range(num_classes):
        class_map[sorted_classes[i]] = i

    print(class_map)
    targets = np.array(class_map)[targets].tolist()

    print(np.array(class_map))
    class_data = [[] for i in range(num_classes)]
    for i in range(len(targets)):
        j = targets[i]
        class_data[j].append(i)

    cls_num_list = [np.sum(np.array(targets)==i) for i in range(num_classes)]
    print(cls_num_list)
    # print(targets)

    plt.bar(range(len(cls_num_list)), cls_num_list)

    plt.show()

# count_instance("datasets/data_txt/Places_LT_train.txt", 365)
# count_instance("datasets/data_txt/ImageNet_LT_train.txt", 1000)
# count_instance("datasets/data_txt/iNaturalist18_train.txt", 8142)

def modify_instance(max_line, mark):
    previous_target = 0
    count = 0
    # with open("datasets/data_txt/Places_BL_train_" + str(max_line) + "_" + str(mark) +  ".txt", 'w') as wf:
    #     with open("datasets/data_txt/Places_LT_train.txt", 'r') as f:
    with open("datasets/data_txt/ImageNet_BL_train_" + str(max_line) + "_" + str(mark) +  ".txt", 'w') as wf:
        with open("datasets/data_txt/ImageNet_LT_train.txt", 'r') as f:

            single = []
            for line in f:
                target = int(line.split()[1])
                if previous_target == target:
                    # count +=1
                    # if count <= 150:
                    #     wf.writelines(line)
                    single.append(line)
                else:
                    previous_target = target
                    random.shuffle(single)
                    if len(single) >= max_line:
                        for i in range(max_line):
                            wf.writelines(single[i])
                    else:
                        for i in single:
                            wf.writelines(i)
                    single = [line]
            
            if len(single) != 0:
                random.shuffle(single)
                if len(single) >= max_line:
                    for i in range(max_line):
                        wf.writelines(single[i])
                else:
                    for i in single:
                        wf.writelines(i)

# modify_instance(10, 1)
# modify_instance(10, 2)
# modify_instance(10, 3)
# modify_instance(10, 4)
# modify_instance(10, 5)
# modify_instance(10, 6)
# modify_instance(10, 7)
# modify_instance(10, 8)
# modify_instance(10, 9)
# modify_instance(10, 0)


# modify_instance(15, 1)
# modify_instance(15, 2)
# modify_instance(15, 3)
# modify_instance(15, 4)
# modify_instance(15, 5)
# modify_instance(15, 6)
# modify_instance(15, 7)
# modify_instance(15, 8)
# modify_instance(15, 9)
# modify_instance(15, 0)

# modify_instance(10, 11)
# modify_instance(10, 12)
# modify_instance(10, 13)
# modify_instance(10, 14)
# modify_instance(10, 15)
# modify_instance(10, 16)
# modify_instance(10, 17)
# modify_instance(10, 18)
# modify_instance(10, 19)
# modify_instance(10, 10)

# modify_instance(5, 1)
# modify_instance(5, 2)
# modify_instance(5, 3)
# modify_instance(5, 4)
# modify_instance(5, 5)
# modify_instance(5, 6)
# modify_instance(5, 7)
# modify_instance(5, 8)
# modify_instance(5, 9)
# modify_instance(5, 0)

# modify_instance(5, 11)
# modify_instance(5, 12)
# modify_instance(5, 13)
# modify_instance(5, 14)
# modify_instance(5, 15)
# modify_instance(5, 16)
# modify_instance(5, 17)
# modify_instance(5, 18)
# modify_instance(5, 19)
# modify_instance(5, 10)

def modify_instance_ina(max_line, mark):
    targets = {}
    with open("datasets/data_txt/iNaturalist18_train.txt", 'r') as f:
        for line in f:
            classes = int(line.split()[1])
            if classes not in targets.keys():
                targets[classes] = [line]
            else:
                targets[classes].append(line)

    with open("datasets/data_txt/iNaturalist18_BL_train_" + str(max_line) + "_" + str(mark) +  ".txt", 'w') as wf:
        for k,v in targets.items():
            random.shuffle(v)
            if len(v) >= max_line:
                for i in range(max_line):
                    wf.writelines(v[i])
            else:
                for i in v:
                    wf.writelines(i)

# modify_instance_ina(10, 1)
# modify_instance_ina(10, 2)
# modify_instance_ina(10, 3)
# modify_instance_ina(10, 4)
# modify_instance_ina(10, 5)
# modify_instance_ina(10, 6)
# modify_instance_ina(10, 7)
# modify_instance_ina(10, 8)
# modify_instance_ina(10, 9)
# modify_instance_ina(10, 0)


def combine_instance():
    source1 = []
    source2 = []

    with open("datasets/data_txt/Places_BL_train_5_0.txt", 'r') as f1:
        for line in f1:
            source1.append(line)

    with open("datasets/data_txt/Places_BL_train_5_1.txt", 'r') as f2:
        for line in f2:
            source2.append(line)

    with open("datasets/data_txt/Places_BL_train_5_0_1.txt", 'w') as wf:
        for k in source1:
            if k not in source2:
                wf.writelines(k)
        for k in source2:
            wf.writelines(k)

combine_instance()

def merge_two_model(ratio):
    root_path = "saved/"
    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_202303081505/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_202303102137/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_202303131936/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303082000/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_202303141602/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_202303081505/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303152130/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_202303081505/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303171628/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_202303081505/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303201903/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303202117/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222000/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222008/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup02.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041243/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041243/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041246/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222000/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222008/ckps/current.pth.tar", map_location='cpu')

    model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_202303081505/ckps/current.pth.tar", map_location='cpu')
    model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222008/ckps/current.pth.tar", map_location='cpu')

    l = list(model_dict2['state_dict_model'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_model'][k].float(), model_dict2['state_dict_model'][k].float()))

    l = list(model_dict2['state_dict_block'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_block'][k].float(), model_dict2['state_dict_block'][k].float()))
        model_dict1['state_dict_block'][k] = ratio * model_dict1['state_dict_block'][k] + (1- ratio) * model_dict2['state_dict_block'][k]

    l = list(model_dict2['state_dict_classifier'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_classifier'][k].float(), model_dict2['state_dict_classifier'][k].float()))
        model_dict1['state_dict_classifier'][k] = ratio * model_dict1['state_dict_classifier'][k] + (1- ratio) * model_dict2['state_dict_classifier'][k]

    # l = list(model_dict2['state_dict_lws_model'].keys())
    # for k in l:
    #     print(k, torch.dist(model_dict1['state_dict_lws_model'][k].float(), model_dict2['state_dict_lws_model'][k].float()))
    #     model_dict1['state_dict_lws_model'][k] = ratio * model_dict1['state_dict_lws_model'][k] + (1- ratio) * model_dict2['state_dict_lws_model'][k]

    torch.save(model_dict1, root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222000_ori_" + str(ratio) + ".pth")

# merge_two_model(0.1)
# merge_two_model(0.2)
# merge_two_model(0.3)
# merge_two_model(0.4)
# merge_two_model(0.5)
# merge_two_model(0.6)
# merge_two_model(0.7)
# merge_two_model(0.8)
# merge_two_model(0.9)


def cal_dis_two_model():
    root_path = "saved/"
    model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_202303141602/ckps/current.pth.tar", map_location='cpu')
    model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_202303081505/ckps/current.pth.tar", map_location='cpu')

    
    l = list(model_dict2['state_dict_model'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_model'][k].float(), model_dict2['state_dict_model'][k].float()))

    l = list(model_dict2['state_dict_block'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_block'][k].float(), model_dict2['state_dict_block'][k].float()))

    l = list(model_dict2['state_dict_classifier'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_classifier'][k].float(), model_dict2['state_dict_classifier'][k].float()))

    l = list(model_dict2['state_dict_lws_model'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_lws_model'][k].float(), model_dict2['state_dict_lws_model'][k].float()))
# cal_dis_two_model()

def merge_mutli_model():
    root_path = "saved/"
    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303201903/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303202117/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303210000/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303210943/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303210944/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303211102/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303211356/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303211404/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303211434/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_202303211448/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222000/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222008/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222022/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222029/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222108/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222109/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222110/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303222112/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303231054/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303231055/ckps/current.pth.tar", map_location='cpu')

    # model_dict11 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241218/ckps/current.pth.tar", map_location='cpu')
    # model_dict12 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241222/ckps/current.pth.tar", map_location='cpu')
    # model_dict13 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241224/ckps/current.pth.tar", map_location='cpu')
    # model_dict14 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241225/ckps/current.pth.tar", map_location='cpu')
    # model_dict15 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241426/ckps/current.pth.tar", map_location='cpu')
    # model_dict16 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241436/ckps/current.pth.tar", map_location='cpu')
    # model_dict17 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241437/ckps/current.pth.tar", map_location='cpu')
    # model_dict18 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241438/ckps/current.pth.tar", map_location='cpu')
    # model_dict19 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241731/ckps/current.pth.tar", map_location='cpu')
    # model_dict20 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_202303241734/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303261940/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303261941/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303261942/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303261944/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303261959/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303262000/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303262001/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303262002/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303262011/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "places_resnet152_stage2_mislas_202303262022/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303301358/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303301400/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303301619/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303301620/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303302000/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303302034/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303302056/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303302128/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303302139/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "places_resnet152_stage2_mislas_bl_5_202303310958/ckps/current.pth.tar", map_location='cpu')


    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012107/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012108/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012110/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012112/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012327/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012329/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012330/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012331/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012333/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012334/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041243/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041246/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041250/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041345/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041348/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041350/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041648/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041702/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041742/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304041746/ckps/current.pth.tar", map_location='cpu')

    # model_dict11 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304131506/ckps/current.pth.tar", map_location='cpu')
    # model_dict12 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304131510/ckps/current.pth.tar", map_location='cpu')
    # model_dict13 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304131614/ckps/current.pth.tar", map_location='cpu')
    # model_dict14 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304131618/ckps/current.pth.tar", map_location='cpu')
    # model_dict15 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304131809/ckps/current.pth.tar", map_location='cpu')
    # model_dict16 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304131817/ckps/current.pth.tar", map_location='cpu')
    # model_dict17 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304132016/ckps/current.pth.tar", map_location='cpu')
    # model_dict18 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304132017/ckps/current.pth.tar", map_location='cpu')
    # model_dict19 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304132057/ckps/current.pth.tar", map_location='cpu')
    # model_dict20 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304132107/ckps/current.pth.tar", map_location='cpu')

    model_dict1 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304151612/ckps/current.pth.tar", map_location='cpu')
    model_dict2 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304151628/ckps/current.pth.tar", map_location='cpu')
    model_dict3 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304151951/ckps/current.pth.tar", map_location='cpu')
    model_dict4 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304151954/ckps/current.pth.tar", map_location='cpu')
    model_dict5 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304152039/ckps/current.pth.tar", map_location='cpu')
    model_dict6 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304152055/ckps/current.pth.tar", map_location='cpu')
    model_dict7 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304152211/ckps/current.pth.tar", map_location='cpu')
    model_dict8 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304161559/ckps/current.pth.tar", map_location='cpu')
    model_dict9 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304161651/ckps/current.pth.tar", map_location='cpu')
    model_dict10 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_202304161953/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304091337/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304091341/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304091342/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304091343/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304091632/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304092139/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304092346/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304092347/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304092348/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_202304092350/ckps/current.pth.tar", map_location='cpu')

    # model_dict1 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111016/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111019/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111020/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111037/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111107/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111108/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111110/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111424/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111427/ckps/current.pth.tar", map_location='cpu')
    # model_dict10 = torch.load(root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_202304111429/ckps/current.pth.tar", map_location='cpu')

    print(model_dict2.keys())
    # l = list(model_dict2['state_dict_model'].keys())
    # for k in l:
    #     print(k, torch.dist(model_dict1['state_dict_model'][k].float(), model_dict2['state_dict_model'][k].float()))

    l = list(model_dict2['state_dict_model'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_model'][k].float(), model_dict2['state_dict_model'][k].float()))
        model_dict1['state_dict_model'][k] = 0.1 * model_dict1['state_dict_model'][k] + 0.1 * model_dict2['state_dict_model'][k] +  \
                                             0.1 * model_dict3['state_dict_model'][k] + 0.1 * model_dict4['state_dict_model'][k] + \
                                             0.1 * model_dict5['state_dict_model'][k] + 0.1 * model_dict6['state_dict_model'][k] + \
                                             0.1 * model_dict7['state_dict_model'][k] +  0.1 * model_dict8['state_dict_model'][k] + \
                                             0.1 * model_dict9['state_dict_model'][k] +  0.1 * model_dict10['state_dict_model'][k] 
        # model_dict1['state_dict_model'][k] = 0.05 * model_dict1['state_dict_model'][k] + 0.05 * model_dict2['state_dict_model'][k] +  \
        #                                      0.05 * model_dict3['state_dict_model'][k] + 0.05 * model_dict4['state_dict_model'][k] + \
        #                                      0.05 * model_dict5['state_dict_model'][k] + 0.05 * model_dict6['state_dict_model'][k] + \
        #                                      0.05 * model_dict7['state_dict_model'][k] +  0.05 * model_dict8['state_dict_model'][k] + \
        #                                      0.05 * model_dict9['state_dict_model'][k] +  0.05 * model_dict10['state_dict_model'][k] + \
        #                                      0.05 * model_dict11['state_dict_model'][k] + 0.05 * model_dict12['state_dict_model'][k] +  \
        #                                      0.05 * model_dict13['state_dict_model'][k] + 0.05 * model_dict14['state_dict_model'][k] + \
        #                                      0.05 * model_dict15['state_dict_model'][k] + 0.05 * model_dict16['state_dict_model'][k] + \
        #                                      0.05 * model_dict17['state_dict_model'][k] +  0.05 * model_dict18['state_dict_model'][k] + \
        #                                      0.05 * model_dict19['state_dict_model'][k] +  0.05 * model_dict20['state_dict_model'][k] 


    # l = list(model_dict2['state_dict_block'].keys())
    # for k in l:
    #     print(k, torch.dist(model_dict1['state_dict_block'][k].float(), model_dict2['state_dict_block'][k].float()))
    #     model_dict1['state_dict_block'][k] = 0.1 * model_dict1['state_dict_block'][k] + 0.1 * model_dict2['state_dict_block'][k] +  \
    #                                          0.1 * model_dict3['state_dict_block'][k] + 0.1 * model_dict4['state_dict_block'][k] + \
    #                                          0.1 * model_dict5['state_dict_block'][k] + 0.1 * model_dict6['state_dict_block'][k] + \
    #                                          0.1 * model_dict7['state_dict_block'][k] +  0.1 * model_dict8['state_dict_block'][k] + \
    #                                          0.1 * model_dict9['state_dict_block'][k] +  0.1 * model_dict10['state_dict_block'][k] 

        # model_dict1['state_dict_block'][k] = 0.05 * model_dict1['state_dict_block'][k] + 0.05 * model_dict2['state_dict_block'][k] +  \
        #                                      0.05 * model_dict3['state_dict_block'][k] + 0.05 * model_dict4['state_dict_block'][k] + \
        #                                      0.05 * model_dict5['state_dict_block'][k] + 0.05 * model_dict6['state_dict_block'][k] + \
        #                                      0.05 * model_dict7['state_dict_block'][k] +  0.05 * model_dict8['state_dict_block'][k] + \
        #                                      0.05 * model_dict9['state_dict_block'][k] +  0.05 * model_dict10['state_dict_block'][k] + \
        #                                      0.05 * model_dict11['state_dict_block'][k] + 0.05 * model_dict12['state_dict_block'][k] +  \
        #                                      0.05 * model_dict13['state_dict_block'][k] + 0.05 * model_dict14['state_dict_block'][k] + \
        #                                      0.05 * model_dict15['state_dict_block'][k] + 0.05 * model_dict16['state_dict_block'][k] + \
        #                                      0.05 * model_dict17['state_dict_block'][k] +  0.05 * model_dict18['state_dict_block'][k] + \
        #                                      0.05 * model_dict19['state_dict_block'][k] +  0.05 * model_dict20['state_dict_block'][k] 

    l = list(model_dict2['state_dict_classifier'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_classifier'][k].float(), model_dict2['state_dict_classifier'][k].float()))
        # model_dict1['state_dict_classifier'][k]=0.125*model_dict1['state_dict_classifier'][k]+0.125* model_dict2['state_dict_classifier'][k] +  \
        #                                      0.125* model_dict3['state_dict_classifier'][k] + 0.125* model_dict4['state_dict_classifier'][k] + \
        #                                      0.125* model_dict5['state_dict_classifier'][k] + 0.125* model_dict6['state_dict_classifier'][k] + \
        #                                      0.125* model_dict7['state_dict_classifier'][k] +  0.125* model_dict8['state_dict_classifier'][k] 

        # model_dict1['state_dict_classifier'][k] = 0.25 * model_dict1['state_dict_classifier'][k]+ 0.25 * model_dict2['state_dict_classifier'][k]+ \
        #                                      0.25 * model_dict3['state_dict_classifier'][k] + 0.25 * model_dict4['state_dict_classifier'][k] 

        model_dict1['state_dict_classifier'][k] = 0.1 * model_dict1['state_dict_classifier'][k]+ 0.1 * model_dict2['state_dict_classifier'][k]+ \
                                             0.1 * model_dict3['state_dict_classifier'][k] + 0.1 * model_dict4['state_dict_classifier'][k] + \
                                             0.1 * model_dict5['state_dict_classifier'][k] + 0.1 * model_dict6['state_dict_classifier'][k] + \
                                             0.1 * model_dict7['state_dict_classifier'][k] +  0.1 * model_dict8['state_dict_classifier'][k] + \
                                             0.1 * model_dict9['state_dict_classifier'][k] +  0.1 * model_dict10['state_dict_classifier'][k] 
        # model_dict1['state_dict_classifier'][k] = 0.05* model_dict1['state_dict_classifier'][k]+ 0.05* model_dict2['state_dict_classifier'][k]+ \
        #                                      0.05* model_dict3['state_dict_classifier'][k] + 0.05* model_dict4['state_dict_classifier'][k] + \
        #                                      0.05* model_dict5['state_dict_classifier'][k] + 0.05* model_dict6['state_dict_classifier'][k] + \
        #                                      0.05* model_dict7['state_dict_classifier'][k] +  0.05* model_dict8['state_dict_classifier'][k] + \
        #                                      0.05* model_dict9['state_dict_classifier'][k] +  0.05* model_dict10['state_dict_classifier'][k] + \
        #                                      0.05* model_dict11['state_dict_classifier'][k]+ 0.05* model_dict12['state_dict_classifier'][k] + \
        #                                      0.05* model_dict13['state_dict_classifier'][k] + 0.05* model_dict14['state_dict_classifier'][k] + \
        #                                      0.05* model_dict15['state_dict_classifier'][k] + 0.05* model_dict16['state_dict_classifier'][k] + \
        #                                      0.05* model_dict17['state_dict_classifier'][k] +  0.05* model_dict18['state_dict_classifier'][k] + \
        #                                      0.05* model_dict19['state_dict_classifier'][k] +  0.05* model_dict20['state_dict_classifier'][k] 
    # l = list(model_dict2['state_dict_lws_model'].keys())
    # for k in l:
    #     print(k, torch.dist(model_dict1['state_dict_lws_model'][k].float(), model_dict2['state_dict_lws_model'][k].float()))
    #     model_dict1['state_dict_lws_model'][k] = 0.1 * model_dict1['state_dict_lws_model'][k]+ 0.1 * model_dict2['state_dict_lws_model'][k]+ \
    #                                          0.1 * model_dict3['state_dict_lws_model'][k] + 0.1 * model_dict4['state_dict_lws_model'][k] + \
    #                                          0.1 * model_dict5['state_dict_lws_model'][k] + 0.1 * model_dict6['state_dict_lws_model'][k] + \
    #                                          0.1 * model_dict7['state_dict_lws_model'][k] +  0.1 * model_dict8['state_dict_lws_model'][k] + \
    #                                          0.1 * model_dict9['state_dict_lws_model'][k] +  0.1 * model_dict10['state_dict_lws_model'][k] 
    # torch.save(model_dict1, root_path +  "places_resnet152_stage2_mislas_5_10dataset_wopretrain.pth")
    # torch.save(model_dict1, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset.pth")
    # torch.save(model_dict1, root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_10dataset_0.005.pth")
    # torch.save(model_dict1, root_path +  "imagenet_resnet50_stage1_mixup_bl_10_classifier_20dataset.pth")
    # torch.save(model_dict1, root_path +  "ina2018_resnet50_stage1_mixup_bl_10_classifier_10dataset.pth")
    # torch.save(model_dict1, root_path +  "places_resnet152_stage1_mixup_bl_5_classifier_20dataset.pth")
    torch.save(model_dict1, root_path +  "imagenet_resnet50_stage1_mixup_bl_5_classifier_10dataset.pth")
# merge_mutli_model()