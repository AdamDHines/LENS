Create a SPECK dataset
===========================
.. important::

    Before proceeding, please ensure you have the `necessary USB permissions <https://synsense-sys-int.gitlab.io/samna/0.45.3/install.html#udev-rules-on-linux-systems>`_.

Collect events for images
-----------------
To create a dataset from the Speck2f Development Kit, you can simply run the following in your command terminal:

.. code-block:: bash
    
    pixi run speck_collect

An instance of the `samnsagui <https://synsense-sys-int.gitlab.io/samna/0.45.3/gui/index.html>`_ will open and show you the events being collected.

Once started, begin collecting your dataset by moving through the environment. To finish collecting, simply close the samnagui window.

Our dataset generator script will automatically create and save images and create the necessary ``.csv`` file for training your model.

Modify collection window
-----------------
By default, the dataset generator will collect over a 1-second timewindow. If we want to increase or decrease the timewindow, we can
use the ``--timebin`` argument to specify the number of milliseconds to collect over:

.. code-block:: bash

    pixi run speck_collect --timebin 250

This will collect over 250 ms, instead of the default 1,000 ms.

Set the experimental details
-----------------
When creating the dataset, you can specify the dataset name, camera, and experiment name. Let's for example say we're collecting an outdoor
dataset called ``OutdoorNav``, on the in-built ``davis128``, and we want to call it ``traverse001``:

.. code-block:: bash

    pixi run speck_collect --dataset OutdoorNav --camera davis128 --data_name traverse001

This will create and store all the images in the ``./lens/dataset/OutdoorNav/davis128/traverse001/`` and generate the necessary ``.csv`` file
for model training.

Then, if we wanted to train on this dataset we'd simply need to run:

.. code-block:: bash

    pixi run train --dataset OutdoorNav --camera davis128 --reference traverse001 --reference_places <NUM_REFERENCES>

Dataset considerations
-----------------
For SPECK datasets, and indeed any datasets being deployed to the device, must consider strict model size constraints for place encoding.

.. warning::

    Model sizes for LENS deployed on SPECK are limited to approximately 180 kB **in total** due to the available memory in each
    neurocore.

    Refer to the `Sinabs documentation <https://sinabs.readthedocs.io/en/v2.0.0/speck/overview.html#memory-constraints-and-network-sizing>`_ for
    more information.

In general, a network architecure of 100 input neurons, 200 feature neurons, and 100 output neurons is roughly the limit for SPECK datasets.

.. hint::

    Although larger input and feature sizes limit the number of places to encode, we have succesfully learned over 600 places on SPECK using
    smaller input and feature encodings.

    See our `paper <https://www.science.org/doi/10.1126/scirobotics.ads3968>`_ for more information.