from setuptools import setup

setup(
    name='biberplus',
    version='0.3.0',
    description='A pure Python implementation of Biber\'s (1988) Variation across Speech and Writing linguistic '
                'tags with additional features',
    url='https://github.com/davidjurgens/biberplus',
    author='Kenan Alkiek, David Jurgens',
    author_email='kalkiek@umich.edu',
    license='MIT License',
    packages=['biberplus', 'biberplus.tagger', 'biberplus.reducer'],
    py_modules=['biberplus', 'biberplus.tagger', 'biberplus.reducer'],
    include_package_data=True,
    package_data={'': ['tagger/constants/*.txt', 'tagger/config.yaml']},
    install_requires=['pandas', 'numpy', 'spacy', 'tqdm',
                      'blis', 'confection', 'PyYAML', 'factor_analyzer',
                      'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl'],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
)
