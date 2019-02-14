from setuptools import setup, find_packages
import os

def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='superreg',
      version='0.0.1',
      description='Super resolution image registration',
      long_description=read('README.md'),
      keywords='inference registration image super-resolution',
      url='',
      author='Colin Clement',
      author_email='colin.clement@gmail.com',
      license='GPLv3',
      packages=find_packages(),  # exclude=['test*']),
      install_requires=['pyfftw', 'scipy', 'numpy'],
      python_requires='>=2.7, <4',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ]
     )
