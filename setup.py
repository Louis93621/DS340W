import os
import subprocess
import sys
from pathlib import Path
from typing import Union

from setuptools import find_packages, setup


def get_sha(root_dir: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root_dir)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return 'UNKNOWN'


def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def main():
    package_name = 'yolov9-vx'
    submodule_name = 'models'
    install_reqs = parse_requirements("./requirements.txt")
    print(os.path.join(os.path.dirname(__file__), submodule_name))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), submodule_name))
    import models

    package_version = models.__version__
    if package_version == 'REPLACE':
        default_version = '0.0.1'
        sha = get_sha(Path.cwd())
        if sha == 'UNKNOWN':
            package_version = f'{default_version}+dev'
        else:
            package_version = f'{default_version}+git{sha[:7]}'

    setup(name=package_name,
          version=package_version,
          install_requires=[install_reqs],
          description='A python package to run YOLOv9 models (non Ultralytics)',
          url='https://github.com/vexcel-data/yolov9.git',
          author='ML team',
          python_requires='>=3.6,<4',
          packages=find_packages(include=['models', 'utils']),
          py_modules=["yolov9_vx"],
          )        


if __name__ == '__main__':
    # Make sure that wheel package is installed python3 -m pip install wheel
    # python3 setup.py bdist_wheel
    main()