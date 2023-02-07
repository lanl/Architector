from setuptools import setup
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='architector',
    version=versioneer.get_version(),
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
        'mendeleev',
        'sqlalchemy<2.0.0'
    ],
    license="BSD 3-Clause License",
    classifiers=["Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry"],
    description="The architector python package - for 3D inorganometallic complex design.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass=versioneer.get_cmdclass(),
)
