from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "binding",
        [],  # sources为空，因为CMake构建
    ),
]

setup(
    name="cuda-algo-lab",
    version="0.0.1",
    author="yami",
    description="CUDA vs Python Algo Lab",
    packages=["python"],  # 让 python/ 被识别为 package
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
