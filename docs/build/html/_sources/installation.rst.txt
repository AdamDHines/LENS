Installation
=================

Packages and dependencies
-----------------

Pixi
^^^^^^^^^^^^^^^^^
.. note::

    This is the recommended method for installing LENS packages and dependencies.

For simplicity and ease of use, we use and highly recommend `pixi <https://prefix.dev/>`_ for package and dependency management. If you have not already 
installed pixi, please run the following in your command terminal:

.. code-block:: bash
    
    curl -fsSL https://pixi.sh/install.sh | bash

The provided ``pixi.toml`` and ``pixi.lock`` files in the repository handle all packages and dependencies and the operation of all the various functions in LENS.
This documentation focuses on using pixi to run evaluation and training scripts.

.. note::

    You will be prompted to restart your terminal after installation, please do so before proceeding. For more information, 
    visit the `pixi documentation <https://pixi.sh/latest/>`_.

Conda
^^^^^^^^^^^^^^^^^
Packages and dependencies can also be installed in a conda environment using our maintained `conda-forge package <https://anaconda.org/conda-forge/lens-vpr>`_. 
We recommend installing and using `micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_ instead of conda by
running the following in your command terminal:

.. code-block:: bash

    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

Once installed, create a new micromamba environment and install our conda-forge package by running the following in your command terminal:

.. code-block:: bash

    micromamba create -n lens -c conda-forge lens-vpr
    micromamba activate lens
    pip install samna

The ``pip install samna`` is necessary since this package is not available on conda-forge.

PyPi
^^^^^^^^^^^^^^^^^
.. warning::

    We do not recommend using pip to install dependencies for LENS, please use pixi.

All the packages and dependencies can alternatively be installed from `PyPi <https://pypi.org/project/lens-vpr/>`_. Before installing,
you will need to ensure that your Python version is <= 3.11.

.. code-block:: bash

    pip install lens-vpr

Repository cloning
-----------------
To get the latest release of LENS, ensure you have `git <https://git-scm.com/downloads>`_ installed and clone the repository by running the following
in your terminal:

.. code-block:: bash

    git clone git@github.com:AdamDHines/LENS.git

Then navigate to the LENS project directory:

.. code-block:: bash

    cd LENS