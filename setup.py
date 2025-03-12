from setuptools import setup, find_packages

setup(
    name='pamseek',  # The name of your package
    version='0.1.0',  # Package version
    description='A package for underwater PAM data analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yang Y',
    author_email='proyangy@gmail.com',
    url='https://github.com/yangyangMarine/pamseek', 
    packages=find_packages(), 
    install_requires=[ 
        'numpy',
        'matplotlib',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9', 
)


def read_requirements(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    _requires = []
    _links = []
    for line in lines:
        if line.startswith("git+"):
            _links.append(line)
        else:
            _requires.append(line)
    return _requires, _links


install_requires, dependency_links = read_requirements('requirements.txt')

setup(install_requires=install_requires)
