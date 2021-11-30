from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Self-descriptive entries which should always be present
    name='ODBO',
    author='Lixue Cheng',
    author_email='lcheng2@caltech.edu',
    url="https://github.com/sherrylixuecheng/ODBO",
    license='Open Source',

    # What packages are required for install
    install_requires=[],
    extras_require={
        'tests': [
            'unittest',
        ],
    },
    packages=["odbo"],
)
