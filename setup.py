from distutils.core import setup
from distutils.extension import Extension
from sys import platform

import numpy
from Cython.Distutils import build_ext

exec(open('version.py').read())

set_dist_ext = Extension("rrpcd.relkern.cy_set_dist",
                         sources=['rrpcd/relkern/cy_set_dist.pyx', 'rrpcd/relkern/c_cy_set_dist.cpp', 'rrpcd/relkern/c_cy_simple_dist.cpp', 'rrpcd/relkern/Assignment.cpp',
                                  'rrpcd/relkern/BipartiteGraph.cpp', 'rrpcd/relkern/Hungarian.cpp'],
                         language='c++',
                         extra_compile_args=["-std=c++11", "-stdlib=libc++",
                                             "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                         extra_link_args=["-std=c++11", "-stdlib=libc++",
                                          "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                         include_dirs=[numpy.get_include()])

setup(
    name='rrpcd',
    packages=['rrpcd', 'rrpcd._UAI_2019', 'rrpcd.experiments', 'rrpcd.relkern', 'rrpcd.tests'],
    version=__version__,
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com',

    cmdclass={'build_ext': build_ext},
    ext_modules=[set_dist_ext],

)
#  python3 setup.py build_ext --inplace
#  pip install -e .
