_base_ = [
    '../_base_/models/rednet38.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs2048_coslr_130e.py',
    '../_base_/default_runtime.py'
]
