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


def merge_mutli_model():
    root_path = "pretrained/places_resnet152_stage1_mixup_bl/"

    model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012107/ckps/current.pth.tar", map_location='cpu')
    model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012108/ckps/current.pth.tar", map_location='cpu')
    model_dict3 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012110/ckps/current.pth.tar", map_location='cpu')
    model_dict4 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012112/ckps/current.pth.tar", map_location='cpu')
    model_dict5 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012327/ckps/current.pth.tar", map_location='cpu')
    model_dict6 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012329/ckps/current.pth.tar", map_location='cpu')
    model_dict7 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012330/ckps/current.pth.tar", map_location='cpu')
    model_dict8 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012331/ckps/current.pth.tar", map_location='cpu')
    model_dict9 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012333/ckps/current.pth.tar", map_location='cpu')
    model_dict10 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012334/ckps/current.pth.tar", map_location='cpu')


    print(model_dict2.keys())
    r = [0.1] * 10

    l = list(model_dict2['state_dict_model'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_model'][k].float(), model_dict2['state_dict_model'][k].float()))
        model_dict1['state_dict_model'][k] = r[0] * model_dict1['state_dict_model'][k] + r[1] * model_dict2['state_dict_model'][k] +  \
                                             r[2] * model_dict3['state_dict_model'][k] + r[3] * model_dict4['state_dict_model'][k] + \
                                             r[4] * model_dict5['state_dict_model'][k] + r[5] * model_dict6['state_dict_model'][k] + \
                                             r[6] * model_dict7['state_dict_model'][k] +  r[7] * model_dict8['state_dict_model'][k] + \
                                             r[8] * model_dict9['state_dict_model'][k] +  r[9] * model_dict10['state_dict_model'][k] 


    l = list(model_dict2['state_dict_block'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_block'][k].float(), model_dict2['state_dict_block'][k].float()))
        model_dict1['state_dict_block'][k] = r[0] * model_dict1['state_dict_block'][k] + r[1] * model_dict2['state_dict_block'][k] +  \
                                             r[2] * model_dict3['state_dict_block'][k] + r[3] * model_dict4['state_dict_block'][k] + \
                                             r[4] * model_dict5['state_dict_block'][k] + r[5] * model_dict6['state_dict_block'][k] + \
                                             r[6] * model_dict7['state_dict_block'][k] +  r[7] * model_dict8['state_dict_block'][k] + \
                                             r[8] * model_dict9['state_dict_block'][k] +  r[9] * model_dict10['state_dict_block'][k] 

    l = list(model_dict2['state_dict_classifier'].keys())
    for k in l:
        print(k, torch.dist(model_dict1['state_dict_classifier'][k].float(), model_dict2['state_dict_classifier'][k].float()))

        model_dict1['state_dict_classifier'][k] = r[0] * model_dict1['state_dict_classifier'][k]+ r[1] * model_dict2['state_dict_classifier'][k]+ \
                                             r[2] * model_dict3['state_dict_classifier'][k] + r[3] * model_dict4['state_dict_classifier'][k] + \
                                             r[4] * model_dict5['state_dict_classifier'][k] + r[5] * model_dict6['state_dict_classifier'][k] + \
                                             r[6] * model_dict7['state_dict_classifier'][k] +  r[7] * model_dict8['state_dict_classifier'][k] + \
                                             r[8] * model_dict9['state_dict_classifier'][k] +  r[9] * model_dict10['state_dict_classifier'][k] 
    torch.save(model_dict1, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset.pth")
merge_mutli_model()

def merge_two_model_ratio(model1, model2, ratio):

    l = list(model2['state_dict_model'].keys())
    for k in l:
        print(k, torch.dist(model1['state_dict_model'][k].float(), model2['state_dict_model'][k].float()))
        model1['state_dict_model'][k] = ratio * model1['state_dict_model'][k] + (1-ratio) * model2['state_dict_model'][k] 


    l = list(model2['state_dict_block'].keys())
    for k in l:
        print(k, torch.dist(model1['state_dict_block'][k].float(), model2['state_dict_block'][k].float()))
        model1['state_dict_block'][k] = ratio * model1['state_dict_block'][k] + (1-ratio) * model2['state_dict_block'][k]

    l = list(model2['state_dict_classifier'].keys())
    for k in l:
        print(k, torch.dist(model1['state_dict_classifier'][k].float(), model2['state_dict_classifier'][k].float()))

        model1['state_dict_classifier'][k] = ratio * model1['state_dict_classifier'][k]+ (1-ratio) * model2['state_dict_classifier'][k]

    return model1

def model_scoup():  
    root_path = "pretrained/places_resnet152_stage1_mixup_bl/"

    # model_dict1 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012107/ckps/current.pth.tar", map_location='cpu')
    # model_dict2 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012108/ckps/current.pth.tar", map_location='cpu')
    # model_dict3 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012110/ckps/current.pth.tar", map_location='cpu')
    # model_dict4 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012112/ckps/current.pth.tar", map_location='cpu')
    # model_dict5 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012327/ckps/current.pth.tar", map_location='cpu')
    # model_dict6 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012329/ckps/current.pth.tar", map_location='cpu')
    # model_dict7 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012330/ckps/current.pth.tar", map_location='cpu')
    # model_dict8 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012331/ckps/current.pth.tar", map_location='cpu')
    # model_dict9 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012333/ckps/current.pth.tar", map_location='cpu')
    model_dict10 = torch.load(root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_202304012334/ckps/current.pth.tar", map_location='cpu')

    # ori_models = 1
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict1, model_dict2, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_12soup_gmean.pth")

    # model_dict12 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_12soup_gmean.pth")
    # ori_models = 2
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict12, model_dict3, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123soup_gmean.pth")


    # model_dict123 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123soup_gmean.pth")
    # ori_models = 3
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict123, model_dict4, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_1234soup_gmean.pth")

    # model_dict1234 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_1234soup_gmean.pth")
    # ori_models = 4
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict1234, model_dict5, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_12345soup_gmean.pth")


    # model_dict12345 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_12345soup_gmean.pth")
    # ori_models = 5
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict12345, model_dict6, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123456soup_gmean.pth")

    # model_dict123456 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123456soup_gmean.pth")
    # ori_models = 6
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict123456, model_dict7, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_1234567soup_gmean.pth")

    # model_dict123456 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123456soup_gmean.pth")
    # ori_models = 6
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict123456, model_dict8, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_1234568soup_gmean.pth")

    # model_dict123456 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123456soup_gmean.pth")
    # ori_models = 6
    # ratio = ori_models / (ori_models + 1)
    # model = merge_two_model_ratio(model_dict123456, model_dict9, ratio)
    # torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_1234569soup_gmean.pth")

    model_dict123456 = torch.load( root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_123456soup_gmean.pth")
    ori_models = 6
    ratio = ori_models / (ori_models + 1)
    model = merge_two_model_ratio(model_dict123456, model_dict10, ratio)
    torch.save(model, root_path +  "places_resnet152_stage1_mixup_bl_10_classifier_10dataset_12345610soup_gmean.pth")

# model_scoup()