# involution

## Image Classification on ImageNet


## Object Detection and Instance Segmentation on COCO

### Faster R-CNN
|    Backbone     |     Neck    |  Style  | Lr schd | box AP | Config | Download |
| :-------------: | :---------: | :-----: | :-----: | :----: | :------: | :--------: |
|    R-50-FPN     | convolution | pytorch |   1x    | 39.5   | [config]() | [model]() &#124; [log]() |
|    R-50-FPN     |  involution | pytorch |   1x    | 40.2   | [config]() | [model]() &#124; [log]() |

### Mask R-CNN
|    Backbone     |     Neck    |  Style  | Lr schd | box AP | mask AP | Config | Download |
| :-------------: | :---------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | convolution | pytorch |   1x    | 39.9   | 35.7    |  [config]() | [model]() &#124; [log]() |
|    R-50-FPN     |  involution | pytorch |   1x    | 40.8   | 36.4    |  [config]() | [model]() &#124; [log]() |

### RetinaNet
|    Backbone     |     Neck    |  Style  | Lr schd | box AP | Config | Download |
| :-------------: | :---------: | :-----: | :-----: | :----: | :------: | :--------: |
|    R-50-FPN     | convolution | pytorch |   1x    | 38.2   | [config]() | [model]() &#124; [log]() |
|    R-50-FPN     |  involution | pytorch |   1x    | 38.2   | [config]() | [model]() &#124; [log]() |


## Semantic Segmentation on Cityscapes

| Method | Backbone | Neck | Crop Size | Lr schd | mIoU  |                                                                                                                                                                               download                                                                                                                                                                               |
|--------|----------|------|-----------|--------:|------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FPN    | RedNet-50     | convolution | 512x1024  |   80000 | 78.01 | [model]() &#124; [log]() |
| FPN    | RedNet-50     |  involution | 512x1024  |   80000 | 79.13 | [model]() &#124; [log]() |
| UPerNet| RedNet-50     | convolution | 512x1024  |   80000 | 80.55 | [model]() &#124; [log]() |
