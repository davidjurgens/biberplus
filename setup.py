from setuptools import setup

setup(
    name='bibermda',
    version='0.1.0',
    description='A pure Python implementation of Biber\'s (1988) Variation across Speech and Writing linguistic tags',
    url='https://github.com/davidjurgens/biber-multidimensional-register-analysis',
    author='Kenan Alkiek, David Jurgens',
    author_email='kalkiek@umich.edu',
    license='MIT License',
    packages=['bibermda', 'bibermda.analyzer', 'bibermda.tagger'],
    py_modules=['bibermda', 'bibermda.analyzer', 'bibermda.tagger'],
    package_data={'bibermda.tagger': ['bibermda/tagger/constants/*.txt'],
                  'bibermda.analyzer': ['bibermda/tagger/constants/*.txt']},
    install_requires=['pandas', 'numpy', 'spacy', 'tqdm',
                      'blis', 'confection',
                      'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl'],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
)
