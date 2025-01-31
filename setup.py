from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# 'xtb', # Xtb as install requirement is broken, installs correctly through conda.
setup(
    name='architector',
    version='0.1.0',
    author='Michael G. Taylor et al.',
    packages=['architector'],
    package_data={"": ["data/*.csv"]},
    install_requires=[
        'ase',
        'numpy',
        'py3Dmol',
        'pynauty',
        'scipy',
        'pandas',
        'mendeleev'
    ],
    license="BSD 3-Clause License",
    classifiers=["Development Status :: 4 - Beta",
                 "Intended Audience :: Science/Research",
                 "Programming Language :: Python :: 3",
                 "Topic :: Scientific/Engineering :: Chemistry"],
    description="The architector python package - for 3D Coordination Complex Design.",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
