from setuptools import setup

setup(name='mayhem',
      version='0.1',
      description='Deconstructing and then optimizing neural network models',
      url='https://github.com/kumasento/mayhem',
      author='Ruizhe Zhao',
      author_email='vincentzhaorz@gmail.com',
      license='MIT',
      packages=['mayhem'],
      install_requires=[
          'numpy'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

