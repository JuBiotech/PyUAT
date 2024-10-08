#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

test_requirements = ["pytest>=3", "wget"]

setup(
    author="Johannes Seiffarth",
    author_email="j.seiffarth@fz-juelich.de",
    python_requires=">=3.6",
    classifiers=[],
    description="Uncertainty-Aware Tracking Framework.",
    entry_points={
        "console_scripts": [],
    },
    install_requires=requirements,
    extras_require={
        "gurobi": ["gurobipy>=5.9.3"],
    },
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="uat",
    name="uat",
    packages=find_packages(include=["uat", "uat.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/JojoDevel/uat",
    version="0.0.1",
    zip_safe=False,
)
