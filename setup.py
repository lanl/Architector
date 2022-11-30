from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='architector',
    version='0.0.1',
    author='Michael G. Taylor et al.',
    packages=['architector'],
    install_requires=[
        'requests',
        'importlib; python_version >= "3.6"',
        'ase',
        'numpy',
        'py3Dmol',
        'pynauty',
        'scipy',
        'pandas',
        'xtb-python'
    ],
    license="BSD 3-Clause License",
    classifiers=["Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry"],
    description="The architector python package - for 3D inorganometallic complex design.",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
