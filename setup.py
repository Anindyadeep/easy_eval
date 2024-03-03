from setuptools import setup, find_packages

setup(
    name='easy_evaluator',
    version='0.1.1',
    description='A library for easy evaluation of language models',
    author='Anindyadeep',
    author_email='proanindyadeep@gmail.com',
    url='https://github.com/Anindyadeep/easy_eval',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here from requirements.txt
        "fastapi==0.108.0",
        "lm-eval==0.4.1"
    ], 
    include_package_data=True,
)
