import os, sys

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# define the base requires needed for the repo
requirements = [ 
		'torch>=2.1.1',
		'torchvision>=0.16.1',
		'numpy>=1.26.2',
		'pandas>=2.1.1',
		'tqdm>=4.65.0',
		'prettytable>=3.5.0',
		'scikit-learn>=1.2.2',
		'sinabs>=2.0.0',
		'h5py>=3.10.0',
		'imageio>=2.34.1',
		'matplotlib>=3.8.2',
		'pynmea2>=1.19.0',
		'scipy>=1.11.4',
		'seaborn>=0.13.2',
		'wandb>=0.16.2'  
                ]

# define the setup
setup(
    name="lens-vpr",
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
