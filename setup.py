from setuptools import setup, find_packages


# TODO: add dependencies
setup(
    name='PhD_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[], # TODO: Add dependencies
    test_suite='nose.collector',
    tests_require=['nose'],
)