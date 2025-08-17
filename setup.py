
from setuptools import setup, find_packages

setup(
    name="estat",
    version="0.1.0",
    description="A package for estat municipal data processing",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.0.0",
        "openpyxl>=3.0.0",
        "requests>=2.0.0",
        "pytest>=6.0.0",
        "polars>=0.19.0",
        "pyarrow>=6.0.0",
        "tqdm>=4.0.0"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    test_suite="tests",
)