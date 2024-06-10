from setuptools import find_namespace_packages, setup

setup(
    name="pypopsyn",
    packages=find_namespace_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
