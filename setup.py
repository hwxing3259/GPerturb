from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'GPerturb'
LONG_DESCRIPTION = 'Python implementation of GPerturb'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="GPerturb",
    version=VERSION,
    author="Hanwen Xing",
    author_email="<hanwen.xing@wrh.ox.ac.uk>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'first package'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)