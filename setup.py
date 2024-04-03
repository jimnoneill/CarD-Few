from setuptools import setup, find_packages

setup(
    name='CarD-Few',
    version='0.1.0',
    author='Your Name',
    author_email='joneilliii@sdsu.ed',
    description='CarD-Few: Context classifier for Carcinogen Detection by Few-Shot Learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/CarD-Few',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.0.0',
        'datasets>=1.0.0',
        'pandas',
        'numpy',
        'scikit-learn',
        'sentence-transformers',
        'setfit',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={'card_few': ['data/*.tsv']},
)
