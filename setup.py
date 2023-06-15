from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='ablang-train',
    version='0.0.1',
    description='',
    license='BSD 3-clause license',
    maintainer='Tobias Hegelund Olsen',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=find_packages(include=('ablang_train', 'ablang_train.*')),
    package_data={
        '': ['*.txt']
    },
    install_requires=[
        'torch>1.9',
        'pytorch-lightning',
        'scikit-learn',
        'seaborn',
        'requests',
        'einops',
        'rotary-embedding-torch',
        'neptune-client',
        'pandas',
    ],
)
