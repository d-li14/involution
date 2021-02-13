# involution

## Getting Started

This repository is fully based on the [OpenMMLab](https://openmmlab.com/) toolkits. For each individual task, the config and model files follow the same directory organization as [mmcls](https://github.com/open-mmlab/mmclassification), [mmdet](https://github.com/open-mmlab/mmdetection), and [mmseg](https://github.com/open-mmlab/mmsegmentation), so just copy-and-paste them to the corresponding locations to get started.

For more guidance in detail, please refer to the original [mmcls](https://github.com/open-mmlab/mmclassification), [mmdet](https://github.com/open-mmlab/mmdetection), and [mmseg](https://github.com/open-mmlab/mmsegmentation) tutorials.

## Model Zoo

The parameters/FLOPs&#8595; and performance&#8593; are marked in the parentheses. Part of these checkpoints are obtained in our reimplementation runs, whose performance may show slight differences with those reported in our paper.

### Image Classification on ImageNet


Before finetuning on the following downstream tasks, download the ImageNet pre-trained weight and set the `pretrained` argument in `det/configs/_base_/models/*.py` or `seg/configs/_base_/models/*.py` to your local path.

### Object Detection and Instance Segmentation on COCO

#### Faster R-CNN
|    Backbone     |     Neck    |  Style  | Lr schd | Params(M) | Flops(G) | box AP | Config | Download |
| :-------------: | :---------: | :-----: | :-----: |:---------:|:--------:| :----: | :------: | :--------: |
|    R-50-FPN     | convolution | pytorch |   1x    | 31.6<sub>(23.9%&#8595;)</sub> | 177.9<sub>(14.1%&#8595;)</sub> | 39.5<sub>(1.8&#8593;)</sub>   | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/faster_rcnn_red50_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ESOJAF74jK5HrevtBdMDku0Bgf71nC7F4UcMmGWER5z1_w?e=qGPdA5) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ESYSpzei_INMn1wu5qa0Su8B9YxXf_rOtib5xHjb1y2alA?e=Qn3lyd) |
|    R-50-FPN     |  involution | pytorch |   1x    | 29.5<sub>(28.9%&#8595;)</sub> | 135.0<sub>(34.8%&#8595;)</sub> | 40.2<sub>(2.5&#8593;)</sub>   | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/faster_rcnn_red50_neck_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EV90stAJIXxEnDRe0QM0lvwB_jm9jwqwHoBOVVOqosPHJw?e=0QoikN) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/Ec8z-SZbJTxJrAJ3FLq0PSsB1Q7T1dXLvhfHmegQqH7rqA?e=5O9jDY) |

#### Mask R-CNN
|    Backbone     |     Neck    |  Style  | Lr schd | Params(M) | Flops(G) | box AP | mask AP | Config | Download |
| :-------------: | :---------: | :-----: | :-----: |:---------:|:--------:| :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | convolution | pytorch |   1x    | 34.2<sub>(22.6%&#8595;)</sub> | 224.2<sub>(11.5%&#8595;)</sub> | 39.9<sub>(1.5&#8593;)</sub>   | 35.7<sub>(<font color="#008000">0.8&#8593;</font>)</sub>    |  [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/mask_rcnn_red50_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EdheYm71X2pFu427_557zqcBmuKaLKEoU5R0Z2Kwo2alvg?e=qXShyW) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EQK-5qH_XxhHn4QnxmQbJ4cBL3sz9HqjS0EoybT2s1751g?e=4gpwK2) |
|    R-50-FPN     |  involution | pytorch |   1x    | 32.2<sub>(27.1%&#8595;)</sub> | 181.3<sub>(28.5%&#8595;)</sub> | 40.8<sub>(2.4&#8593;)</sub>   | 36.4<sub>(<font color="#008000">1.3&#8593;</font>)</sub>    |  [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/mask_rcnn_red50_neck_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EYYgUzXjJ3VBrscng-5QW_oB9wFK-dcqSDYB-LUXldFweg?e=idFEgd) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ETWdfYuhjY5AlGkUH11rLl4BLk9zsyKgwAbay47TYzIU-w?e=6ey6cD) |

#### RetinaNet
|    Backbone     |     Neck    |  Style  | Lr schd | Params(M) | Flops(G) | box AP | Config | Download |
| :-------------: | :---------: | :-----: | :-----: |:---------:|:--------:| :----: | :------: | :--------: |
|    R-50-FPN     | convolution | pytorch |   1x    | 27.8<sub>(26.3%&#8595;)</sub> | 210.1<sub>(12.2%&#8595;)</sub> | 38.2<sub>(1.6&#8593;)</sub>   | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/retinanet_red50_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EfUY9orEyCVCsYMlcDhIZ2wBBDw7k1HqfTm9u11KfTopmA?e=4Jhu79) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EQQ_EVDmVg1FlfgpAu9NF5wB6xe6qnqaYWKJw9lL7kRxdw?e=fXxjPg) |
|    R-50-FPN     |  involution | pytorch |   1x    | 26.3<sub>(30.2%&#8595;)</sub> | 199.9<sub>(16.5%&#8595;)</sub> | 38.2<sub>(1.6&#8593;)</sub>   | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/retinanet_red50_neck_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EedZ3bMWZkJIvKjyLkTZHksBc_8wdOMHhFZA7RDewjPO8g?e=jsSjYI) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ES7chxQh5-lGr5--GqroMScBKNTNACyvosdVuThPvkZGkg?e=CrlN9F) |


### Semantic Segmentation on Cityscapes

| Method | Backbone | Neck | Crop Size | Lr schd | Params(M) | Flops(G) | mIoU  | Config |                                                                                                                                                                               download                                                                                                                                                                               |
|--------|----------|------|-----------|--------:|:---------:|:--------:|------:|:------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FPN    | RedNet-50     | convolution | 512x1024  |   80000 | 18.5<sub>(35.1%&#8595;)</sub> | 293.9<sub>(19.0%&#8595;)</sub> | 78.0<sub>(3.6&#8593;)</sub> | [config](https://github.com/d-li14/involution/blob/main/seg/configs/involution/fpn_red50_512x1024_80k_cityscapes.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EYstjiI28SJPohJE54wapFUBW5Wc95Di2Rsh0vf6K79vPw?e=lOvbkZ) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EXdupIgFuAlFuH854wThyXcBQTyL7YhK3wPYcR98rw7PJg?e=MyXx2w) |
| FPN    | RedNet-50     |  involution | 512x1024  |   80000 | 16.4<sub>(42.5%&#8595;)</sub> | 205.2<sub>(43.4%&#8595;)</sub> | 79.1<sub>(4.7&#8593;)</sub> | [config](https://github.com/d-li14/involution/blob/main/seg/configs/involution/fpn_red50_neck_512x1024_80k_cityscapes.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EZzDyESh0ElFp2pIFL1xN70BAj1EyvhFyqi0g7Mp1OZxog?e=F7kZYH) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EXcP_3ujO_1Juj8ap7rqDJ8BWZDCyJL86BWjeZiJ_FfLOw?e=47lvtq) |
| UPerNet| RedNet-50     | convolution | 512x1024  |   80000 | 56.4<sub>(15.1%&#8595;)</sub> | 1825.6<sub>(3.6%&#8595;)</sub> | 80.6<sub>(2.4&#8593;)</sub> | [config](https://github.com/d-li14/involution/blob/main/seg/configs/involution/upernet_red50_512x1024_80k_cityscapes.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/Eb8-frsvSuNAm7qQ6-H2DtEBdACuf-mUOBhvE3YIOiobmA?e=Ibb2cN) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EWhyFAZpxfRBoFi1myoT-RMB6-HeaP7NjSv88YQve4bZkg?e=wC8ccl) |

## Citation
