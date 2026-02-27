from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import subprocess

# Get FFmpeg compile/link flags
def get_pkg_config(lib, flag):
    try:
        return subprocess.check_output(
            ['pkg-config', flag, lib], stderr=subprocess.DEVNULL
        ).decode().strip().split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

ffmpeg_cflags = get_pkg_config('libavcodec libavformat libavutil libswscale', '--cflags')
ffmpeg_libs = get_pkg_config('libavcodec libavformat libavutil libswscale', '--libs')

setup(
    name='compressed_emb',
    ext_modules=[
        CppExtension(
            'compressed_emb',
            ['csrc/compressed_emb.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp'] + ffmpeg_cflags,
            extra_link_args=['-fopenmp'] + ffmpeg_libs,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
