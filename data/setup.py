from setuptools import find_packages, setup

# you should run setup.sh instead of this one, setup.sh will call this package
setup(
    name="acecoder_data",
    version="1.0.0",
    description="",
    author="Wyett (Huaye) Zeng",
    author_email="wyettzeng@gmail.com",
    packages=find_packages(),
    url="https://github.com/TIGER-AI-Lab/AceCoder",  # github
    install_requires=[
        # comment the following if you have CUDA 11.8
        "torch",
        "vllm",
        "xformers",
        # Do not comment any of these:
        "accelerate",
        "datasets",
        "numpy",
        "fire",
        "tqdm",
        "transformers",
        "flash_attn",
        "tqdm",
        "datasets",
        "matplotlib",
        "seaborn",
        "rewardbench",
        "openpyxl",
        "scikit-learn",
    ],
)
