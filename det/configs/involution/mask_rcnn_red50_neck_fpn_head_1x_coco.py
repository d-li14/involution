_base_ = [
    '../_base_/models/mask_rcnn_red50_neck_fpn_head.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x_warmup.py', '../_base_/default_runtime.py'
]
optimizer_config = dict(grad_clip(dict(_delete_=True, max_norm=5, norm_type=2)))
