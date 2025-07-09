from setuptools import setup
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="yolov9-vx",
    version="1.0.0",
    author="YOLOv9 Team",
    author_email="",
    description="YOLOv9 models, utilities and tools package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WongKinYiu/yolov9",
    packages=[
        'yolov9_vx',
        'yolov9_vx.models',
        'yolov9_vx.utils',
        'yolov9_vx.utils.segment',
        'yolov9_vx.utils.panoptic',
        'yolov9_vx.utils.tal',
        'yolov9_vx.utils.loggers',
        'yolov9_vx.utils.loggers.clearml',
        'yolov9_vx.utils.loggers.comet',
        'yolov9_vx.utils.loggers.wandb',
        'yolov9_vx.tools'
    ],
    package_dir={
        'yolov9_vx': 'yolov9_vx',
        'yolov9_vx.models': 'models',
        'yolov9_vx.utils': 'utils',
        'yolov9_vx.utils.segment': 'utils/segment',
        'yolov9_vx.utils.panoptic': 'utils/panoptic',
        'yolov9_vx.utils.tal': 'utils/tal',
        'yolov9_vx.utils.loggers': 'utils/loggers',
        'yolov9_vx.utils.loggers.clearml': 'utils/loggers/clearml',
        'yolov9_vx.utils.loggers.comet': 'utils/loggers/comet',
        'yolov9_vx.utils.loggers.wandb': 'utils/loggers/wandb',
        'yolov9_vx.tools': 'tools'
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'yolov9_vx': ['*.yaml', '*.yml'],
        'yolov9_vx.models': ['**/*.yaml', '**/*.yml'],
        'yolov9_vx.utils': ['**/*.yaml', '**/*.yml'],
        'yolov9_vx.tools': ['**/*.yaml', '**/*.yml', '**/*.ipynb'],
    },
)
