from os.path import abspath, dirname, join
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

INCLUDE_DIR = join(dirname(abspath(__file__)), 'include')

setup(
    name='involution-pytorch',
    version="0.1.0",
    url="https://github.com/shikishima-TasakiLab/Involution-PyTorch",
    license="MIT License",
    author="Junya Shikishima",
    author_email="160442065@ccalumni.meijo-u.ac.jp",
    description="PyTorch Involution",
    packages=find_packages(),
    ext_modules=[
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
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
