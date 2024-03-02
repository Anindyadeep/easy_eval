from setuptools import setup, find_packages

setup(
    name='easy_eval',
    version='0.1.0',
    description='A library for easy evaluation of language models',
    author='Anindyadeep',
    author_email='proanindyadeep@gmail.com',
    url='https://github.com/your_username/easy_eval',
    packages=find_packages(),
    install_requires=[
        "fastapi==0.108.0",
        "lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git"
    ], 
)
