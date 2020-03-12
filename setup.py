"""
Minimal setup file for the fast_matched_filter library for Python packaging.

:copyright:
    William B. Frank 
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call

# Get the long description - it won't have md formatting properly without
# using pandoc though, but that adds another dependency.

setup(name='SeisMgr',
      version='0.0.1',
      description='Seismic preprocessing and data management',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: GPL License',
        'Programming Language :: Python :: 2.7, 3.5, 3.6',
        'Topic :: Seismology'
      ],
      author='William Frank',
      license='GPL',
      packages=['seismgr'],
      install_requires=['numpy', 'scipy'],
      include_package_data=True,
      zip_safe=False,
    )
