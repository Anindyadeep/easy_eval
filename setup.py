import os 
import io 
from setuptools import setup, find_packages

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("ateltasdk", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

setup(
    name='easy_evaluator',
    version='{{VERSION_PLACEHOLDER}}',
    description='A library for easy evaluation of language models',
    author='Anindyadeep',
    author_email='proanindyadeep@gmail.com',
    url='https://github.com/Anindyadeep/easy_eval',
    keywords=['llm', 'evaluation', 'openai'],
    packages=find_packages(),
    install_requires=[
        # List your dependencies here from requirements.txt
        "fastapi==0.108.0",
        "lm-eval==0.4.1"
    ], 
    include_package_data=True,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
