from setuptools import setup, find_packages

setup(
    name='acecoder',
    version='0.0.1',
    description='Official Codes for of "ACECODER: Acing Coder RL via Automated Test-Case Synthesis"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    package_dir={'': 'src'},  # Add this line
    packages=find_packages(where='src'),  # Modify this line
    url='https://github.com/TIGER-AI-Lab/AceCoder',
    install_requires=[
        "transformers",
        "torch",
        "datasets",
        "accelerate",
        "evalplus"
    ],
)



# change it to pyproject.toml
# [build-system]
# python setup.py sdist bdist_wheel
# twine upload dist/*