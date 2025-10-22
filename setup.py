from setuptools import find_namespace_packages, setup

setup(
    name="mlpoppyns",
    packages=find_namespace_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
