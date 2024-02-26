# BTM
**Reviving Undersampling for Long-Tailed Learning**

**Authors**: Hao Yu, Yingxiao Du, Jianxin Wu

[[`arXiv`](https://arxiv.org/pdf/2401.16811.pdf)] [[`bibtex`](#Citation)]


**Introduction**: This repository provides an implementation for the paper: "[Reviving Undersampling for Long-Tailed Learning](https://arxiv.org/pdf/2401.16811.pdf)" based on [MiSLAS](https://github.com/dvlab-research/MiSLAS). *We aim to enhance the accuracy of the worst-performing categories and utilize the harmonic mean and geometric mean to assess the model's performance. We revive the balanced undersampling produces a more equitable distribution of accuracy across categories, and devise a straightforward model ensemble strategy, which does not result in any additional overhead and achieves improved harmonic and geometric mean while keeping the average accuracy.* BTM is a simple, and efficient framework for long-tailed recognition.

## Installation

**Requirements**

* Python 3.8
* torchvision 0.13.0
* Pytorch 1.12.0

**Dataset Preparation**
* [ImageNet-LT](http://image-net.org/index)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
* [Places-LT](http://places2.csail.mit.edu/download.html)

Change the `data_path` in `config/*/*.yaml` accordingly.

## Training

**Stage-1**:

To get a model of Stage-1, you can directly download from [MiSLAS](https://github.com/dvlab-research/MiSLAS), or run:

```
python train_stage1.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup.yaml
```

`DATASETNAME` can be selected from `imagenet`, `ina2018`, and `places`.

`ARCH` can be `resnet50/101/152` for `imagenet`, `resnet50` for `ina2018`, and `resnet152` for `places`, respectively.

**BTM**:

To training a model with undersamping, run:
```
python train_stage1_bl_10_classifier.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup_bl_10_calssifier.yaml
```

Modify Line221 `train_loader = dataset.bl_train_10_0_instance` to `bl_train_10_1_instance`, `bl_train_10_2_instance` etc. for getting different balance-training models.

Then run 
```
python merge.py
```

for getting the fusion model. Modify Line19-28 to the real model checkpoint path. 


**Stage-2**:

To train a model for Stage-2, run:

```
python train_stage2.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage2_mislas.yaml resume /path/to/checkpoint/BTM
```

The saved folder (including logs and checkpoints) is organized as follows.
```
MiSLAS
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```
## Evaluation

To evaluate a trained model, run:

```
python eval.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup.yaml  resume /path/to/checkpoint/stage1
python eval.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage2_mislas.yaml resume /path/to/checkpoint/stage2
```

## <a name="Citation"></a>Citation

```bib
@article{yu2024reviving,
  title={Reviving Undersampling for Long-Tailed Learning},
  author={Yu, Hao and Du, Yingxiao and Wu, Jianxin},
  journal={arXiv preprint arXiv:2401.16811},
  year={2024}
}
```

## Contact

If you have any questions about our work, feel free to contact us through email (Hao Yu: yuh@lamda.nju.edu.cn) or Github issues.
