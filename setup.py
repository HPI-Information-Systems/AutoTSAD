import glob
import os
import shutil
import sys
from pathlib import Path

from setuptools import Command, find_packages, setup

README = (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")
HERE = Path(os.path.dirname(__file__)).absolute()
# get __version__ from timeeval/_version.py
with open(Path("autotsad") / "_version.py") as f:
    exec(f.read())
VERSION: str = __version__  # noqa


def load_dependencies():
    EXCLUDES = ["python", "pip"]
    with open(HERE / "requirements.txt", "r", encoding="UTF-8") as f:
        env = f.readlines()

    def excluded(name):
        return any([excl in name for excl in EXCLUDES])

    deps = [dep for dep in env if not excluded(dep)]
    return deps


class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        files = [".coverage*", "coverage.xml"]
        dirs = [
            "build",
            "dist",
            "*.egg-info",
            "**/__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "**/.ipynb_checkpoints",
        ]
        for d in dirs:
            for filename in glob.glob(d):
                shutil.rmtree(filename, ignore_errors=True)

        for f in files:
            for filename in glob.glob(f):
                try:
                    os.remove(filename)
                except OSError:
                    pass


if __name__ == "__main__":
    setup(
        name="AutoTSAD",
        version=VERSION,
        description="Unsupervised Anomaly Detection System for Univariate Time Series",
        long_description=README,
        long_description_content_type="text/markdown",
        author="Sebastian Schmidl",
        author_email="sebastian.schmidl@hpi.de",
        url="https://github.com/HPI-Information-Systems/AutoTSAD",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        packages=find_packages(exclude=("data", "db", "notebooks", "scripts")),
        install_requires=load_dependencies(),
        python_requires=">=3.8, <=3.11",
        cmdclass={"clean": CleanCommand},
        zip_safe=False,
        entry_points={"console_scripts": ["autotsad=autotsad.__main__:main"]},
    )
