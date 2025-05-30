from setuptools import setup, find_packages

setup(
    name='pyneuromesh',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch', 'numpy', 'gmsh', 'trimesh', 'matplotlib'
    ],
)

