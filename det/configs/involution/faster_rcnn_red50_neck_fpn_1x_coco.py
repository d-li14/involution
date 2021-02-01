_base_ = [
    '../_base_/models/faster_rcnn_red50_neck_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x_warmup.py', '../_base_/default_runtime.py'
]
