from setuptools import setup
from setuptools import find_packages
import os

PATH = os.path.dirname(__file__)
with open(os.path.join(PATH, "README.md"), "r") as rf:
    README = rf.read()

version = "1.0.3"

install_requires = [
    "pyyaml",
    "ipdb",
    "scikit-learn",
    "numpy",
    "torch",
    "setuptools",
    "torch >= 1.2.0",
    "torchvision >= 0.4.0",
    "matplotlib",
    "pillow"
]

setup(
    name="extorch",
    version=version,
    description="An useful extension library of PyTorch.",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="PyTorch",
    author="Junbo Zhao",
    author_email="zhaojunbo2012@sina.cn",
    url="https://github.com/A-LinCui/Extorch",
    license="MIT License",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    package_data={"extorch": ["VERSION"]}
)
