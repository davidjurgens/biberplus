from setuptools import setup, find_packages

setup(
    name="biberplus",
    version="0.3.0",
    description="A pure Python implementation of Biber's (1988) Variation across Speech and Writing linguistic tags with additional features",
    url="https://github.com/davidjurgens/biberplus",
    author="Kenan Alkiek, David Jurgens",
    author_email="kalkiek@umich.edu",
    license="MIT License",
    license_files=["LICENSE"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["tagger/constants/*.txt", "tagger/config.yaml"]},
    install_requires=[
        "pandas",
        "numpy",
        "spacy>=3.0.0",
        "tqdm",
        "blis",
        "confection",
        "PyYAML",
        "factor_analyzer",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
