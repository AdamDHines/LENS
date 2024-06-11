import os, sys

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# define the base requires needed for the repo
requirements = [ 
                'torch',
                'torchvision',
                'numpy',
                'pandas',
                'tqdm',
                'prettytable',
                'scikit-learn',
                'matplotlib'
                'sinabs',
                'samna',
                'opencv-python'    
                ]

# define the setup
setup(
    name="LENS",
    version="0.1.0",
    description='LENS: Locational Encoding with Neuromorphic Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Adam D Hines, Michael Milford and Tobias Fischer',
    author_email='adam.hines@qut.edu.au',
    url='https://github.com/AdamDHines/LENS',
    license='MIT',
    install_requires=requirements,
    python_requires='>=3.6, !=3.12.*',
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
      ],
    packages=find_packages(),
    keywords=['robotics','visual-place-recognition','neuromorphic-computing','spiking-neural-network','dynamic-vision-sensors'],
    scripts=['main.py'],
)
