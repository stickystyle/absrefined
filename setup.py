from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="absrefined",
    version="0.1.0",
    description="AudioBookShelf Chapter Marker Refiner",
    author="Original Author",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "abs-chapter-refiner=absrefined.main:main",
        ],
    },
    python_requires=">=3.7",
) 