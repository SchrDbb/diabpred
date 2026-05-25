from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#") and not line.startswith("pytest")
    ]

setup(
    name="diabpred",
    version="1.0.0",
    author="Dassi Bopda Blondel Christian",
    author_email="dassibopdablondel@gmail.com",
    description=(
        "An open-source ML toolkit for diabetes risk prediction "
        "with reproducible benchmarks."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SchrDbb/diabpred",
    packages=find_packages(exclude=["tests*", "scripts*", "examples*", "docs*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "diabpred-run=scripts.run_experiment:main",
        ]
    },
    include_package_data=True,
)
