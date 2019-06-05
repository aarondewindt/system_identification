from setuptools import setup, find_packages
from distutils.util import convert_path


ver_path = convert_path('system_identification/version.py')
with open(ver_path) as ver_file:
    ns = {}
    exec(ver_file.read(), ns)
    version = ns['version']

setup(
    name='system_identification',
    version=ns['version'],
    description='System identification assignment',
    author='Aaron M. de Windt',
    author_email='',

    install_requires=['numpy',
                      'scipy',
                      "xarray",
                      "matplotlib"
                      ],
    packages=find_packages('.', exclude=["test"]),
    package_data={},
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha'],
    entry_points={
        'console_scripts': [
            'si = system_identification.__main__:main'
        ]
    }
)
