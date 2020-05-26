from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name='mcts_cpp',
    version='0.1.0',
    keywords='mcts module',
    description='a library for mcts',
    license='No License',
    url='',
    author='Xuefeng Huang',
    author_email='icybee@yeah.net',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            name="mcts_cpp",
            sources=["mcts_cpp.pyx"],
            include_dirs=[numpy.get_include(),'./mcts_src'],  # 头文件路径
            library_dirs=[],  # 库文件路径
            libraries=[],
            np_pythran=False,
            runtime_library_dirs=[],  # 运行时所需库文件路径
            depends=['mcts_src/mcts_search.cpp', 'mcts_src/mcts_search.hpp', 'mcts_src/dirichlet.hpp'],  # 编译rect.cpp时依赖的源码
            language='c++',
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-std=c++11"],
        )
    ]

)
