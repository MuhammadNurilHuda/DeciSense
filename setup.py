from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="DeciSense",
    version="0.1.0",
    author="Muhammad Nuril Huda",
    author_email="nurilhuda3333@gmail.com",
    description="Config-driven DS automation with LLM decision layer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MuhammadNurilHuda/DeciSense",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip() 
        for line in open('requirements.txt')
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        "console_scripts": [
            "deci=decisense.cli:cli",
        ],
    },
)
