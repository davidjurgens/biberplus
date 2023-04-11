from setuptools import setup

setup(
    name='bibermda',
    version='0.1.0',
    description='A pure Python implementation of Biber\'s (1988) Variation across Speech and Writing linguistic tags',
    url='https://github.com/davidjurgens/biber-multidimensional-register-analysis',
    author='Kenan Alkiek, David Jurgens',
    author_email='kalkiek@umich.edu',
    license='MIT License',
    packages=['bibermda'],
    install_requires=['pandas', 'numpy', 'spacy', 'tqdm'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
)