import os
from os.path import abspath, dirname, join
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

INCLUDE_DIR = join(dirname(abspath(__file__)), 'include')

EXTENSION = []

if os.getenv('USE_CUDA', '1') == '1':
    EXTENSION.append(
        CUDAExtension(
            name='involution',
            sources=[
                'src/involution2d_cpu.cpp',
                'src/involution2d_cuda.cu',
                'src/pytorch_wrapper.cpp',
            ],
            include_dirs=[
                INCLUDE_DIR
            ],
            extra_compile_args={
                'cxx': ['-DUSE_CUDA', '-O3'],
                'nvcc': ['-O3'],
            }
        )
    )
else:
    EXTENSION.append(
        CppExtension(
            name='involution',
            sources=[
                'src/involution2d_cpu.cpp',
                'src/pytorch_wrapper.cpp',
            ],
            include_dirs=[
                INCLUDE_DIR
            ],
            extra_compile_args=['-O3']
        )
    )

setup(
    name='involution-pytorch',
    version="0.1.0",
    url="https://github.com/shikishima-TasakiLab/Involution-PyTorch",
    license="MIT License",
    author="Junya Shikishima",
    author_email="160442065@ccalumni.meijo-u.ac.jp",
    description="PyTorch Involution",
    packages=find_packages(),
    ext_modules=EXTENSION,
    cmdclass={
        'build_ext': BuildExtension,
    }
)
