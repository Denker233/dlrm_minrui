from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='compressed_emb',
    ext_modules=[
        CppExtension(
            'compressed_emb',
            ['csrc/compressed_emb.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
