Training New Models
==================
In this guide, we explore how to train models using LENS. There are a number of options available to tune the model to best fit your data. 
LENS supports a range of DVS cameras with only minimal parameter adjustments.

Preparing your data
-----------------
LENS trains on event frames, which are created by counting the number of events detected over a specific time window. The ideal time window depends on your collection method. 
Faster robot movement speeds might require a shorter collection period, whereas slower speeds would need to be increase. 

.. hint::

    As a guideline, aim for an average of **50–100 events per pixel** and a 1-second time window as a starting point.

The exact method to extract events and generate event frames will depend on your camera and how the events are stored. An example script that extracts events from a DAVIS346
stored in a rosbag is shown `here <https://github.com/AdamDHines/LENS/blob/main/lens/tools/dvstools.py>`_. 

.. important::

    All event frames must be in 8-bit grayscale format and stored as ``.png`` files.

It is recommended that all data for LENS be placed in the ``./lens/dataset/`` directory. The directory structure should follow this convention:

.. code-block:: text

   --dataset
     |--dataset1
        |--camera1
           |experiment001
           |experiment002
        |--camera2
           |experiment003
           |experiment004

.. note::

    This layout supports running the same dataset across different DVS cameras.

As an example, let's look at the example dataset that we have included in the LENS repository:

.. code-block:: text

   --dataset
     |--example
        |--davis128
           |example-query
           |example-reference

To load images in for both training and evaluation, we use a ``.csv`` file of image names and its index value. To simply create this file, we provide a `simple 
script <https://github.com/AdamDHines/LENS/blob/main/lens/tools/create_data_csv.py>`_ that generates one for you.

.. code-block:: python

   create_csv_from_images('./lens/dataset/example/davis128/example-query', 
                     './lens/dataset/example-query.csv', 
                     gps_path=None)

.. note:: 

   This script includes the ability to read GPS coordinates from a ``.nmea`` file and associate it with image timestamps to assist in ground truth creation. Please see `this script
   <https://github.com/AdamDHines/LENS/blob/main/lens/tools/read_gps.py>`_ for more details.

Ground truth file
-----------------
In order to run the evaluation and obtain matching metrics, a ground truth file is required. The ground truth file is a binary matrix stored as an ``.npy`` file in
your dataset directory:

.. code-block:: text

   --dataset
     |--example
        |--davis128
           |example-query
           |example-reference
           |example-reference_example-query_GT.npy

During evaluation, this file is loaded and used to run Recall@K and Precision-Recall analysis.

For datasets where there is a 1:1 correspondence between queries and reference, the ground truth can simply be:

.. code-block:: python

   import numpy as np

   GT = np.eye(<NUM_REFERENCES>, <NUM_QUERIES>)
   np.save(GT, './lens/dataset/<your_dataset>/<camera>/<reference>_<query>_GT.npy')

.. important::

   The naming of the ground truth file must always be <REFERENCE>_<QUERY>_GT with an underscore between the reference and query names.

For datasets with unequal references and queries, the ground truth will need to be generated from GPS coordinates or manually curated. As this will differ greatly
across datasets, formats, and cameras - it is not easily feasible to have a generalized pipeline for ground truth generation. 

Running the training
-----------------
To run the training network, we simply run the following in the command terminal:

.. code-block:: bash

   pixi run train

The default arguments for the example dataset have already been set in LENS. However, if we wanted to see what this would like with a custom dataset we
can add the arguments in manually:

.. code-block:: bash

   pixi run train --dataset example --camera davis128 --reference example-reference --reference_places 100

The arguments ``--dataset``, ``--camera``, ``--reference``, and ``-reference_places`` are used to tell LENS which dataset you want to train on and how
many references images there are in the dataset.

If we had a different dataset, for example we collected two different event streams (stream001 and stream002) from an outdoor environment (OutDoorEv) on a DAVIS346 with 500 images each,
we would modify the arguments as:

.. code-block:: bash

   pixi run train --dataset OutDoorEv --camera DAVIS346 --reference stream001 --reference_places 500

.. hint::

   However the data is stored in ``./lens/dataset/`` is what you will call in the arguments

If your dataset is in a location other than ``./lens/dataset/``, you can change the root dataset directory path:

.. code-block:: bash

   pixi run train --data_dir <your_dataset_location>/

The last dataset related option available is the ability to skip images in your directory. For example, if you want to increase the physical
distance between places trained you can ignore a set interval of images:

.. code-block:: bash

   pixi run train --filter 2

This will skip every 2nd image in the directory, halving the number of reference images.

Network setup
-----------------
There are a plethora of training parameters that can be tuned for different datasets. In general, the default hyperparameters have been found to generalize
well to multiple different datasets from various event cameras.

.. note::

   Please see :doc:`Training Parameters <train_params>` for a full list of training hyperparameters.

Input layer size
^^^^^^^^^^^^^^^^^
LENS works on the basis of selecting pixels from images using a convolutional kernel to reduce input dimensionality. We can automatically
alter the convolution to allow more or fewer pixels in for downsampling using the ``--dims`` argument:

.. code-block:: bash

   pixi run train --dims [10, 10]

This tells LENS to downsample the image from its input size to a ``10x10`` image, `i.e. 100 pixels`.

The ``--roi_dim`` argument informs LENS of the input dimensionality which runs a check against the downsampled size to make sure it is compatible.

.. code-block:: bash

   pixi run train --roi_dim [80, 80]

This argument tells LENS that the input image is of size ``80x80``.

.. note::

   Currently, LENS only supports square input images and outputs. If using rectangular images, please select a square ROI.

Feature layer size
^^^^^^^^^^^^^^^^^
We can modify the number of neurons in our feature layer which will affect the spatial representation of information of input images. Increasing the number of
neurons generally increases performance, to a point, whilst decreasing the number of neurons decreases performance.

The ``--feature_multiplier`` argument controls how many feature neurons there are relative to the number of pixel encoding input neurons.

.. code-block:: bash

   pixi run train --feature_multiplier 4.0 # for 4x the number of neurons relative to input
   pixi run train --feature_multiplier 0.5 # for 1/2x the number of neurons relative to input

Connection probabilities
^^^^^^^^^^^^^^^^^
Altering the connection probabilities easily allows you to sparsify network connections for both excitatory and inhibitory weights. When a new
model is randomly seeded, it will use the connection probability value to set the desired number of connections:

.. code-block:: bash

   pixi run train --f_exc 0.35 --f_inh 0.75 --o_exc 1.0 --o_inh 1.0

The above will set a 35% excitatory and 75% inhibitory connections from the input to the feature and fully connect the feature to output layers.

.. note::

   Whilst we specify that connections probabilities `sparsify` network weights, they are not true sparse matrices. Instead, a lack of connection
   is represented by a weight of ``0`` and does not contribute to synaptic activity or weight updates during training.

Hyperparameters
-----------------
There are a few hyperparameters that we can modify for network training. In general, the defaults work well for a variety of datasets however
this may not be the case for your dataset.

.. hint::

   Check out the :doc:`Optimizer <optimizer_overview>` documentation for how to tune your network parameters for custom datasets.

Training epochs
^^^^^^^^^^^^^^^^^
The number of epochs is controllable for each layer pair in LENS, meaning you can train the input --> feature and feature --> output differently.
Use the ``--epoch_feat`` and ``--epoch_out`` arguments with an integer value to change how many training iterations there are:

.. code-block:: bash

   pixi run train --epoch_feat 64 --epoch_out 128

This will run 64 epochs for the input --> feature layer and 128 epochs for the feature --> output layer.

Learning rates
^^^^^^^^^^^^^^^^^
In addition to epochs, the learning rates for training each layer pair can be different to account for the number of training iterations. 
The ``--stdp_rate_feat`` and ``--stdp_rate_out`` arguments are used to control the learning rates:


.. code-block:: bash

   pixi run train --stdp_rate_feat 1e-2 --stdp_rate_out 1e-3

Spiking thresholds
^^^^^^^^^^^^^^^^^
Spiking thresholds are used to control spiking activity in individual neurons, with higher threshold values requiring bigger spike amplitudes
to propagate information. 

Spike thresholds are uniformly distributed for each layer in a linspace range from ``low`` to ``high`` and are a learnable parameter during
training:

.. code-block:: bash

   pixi run train --thr_l_feat 0 --thr_h_feat 0.75 --thr_l_out 0 --thr_h_out 0.5

This will set an upper spiking threshold bound of ``0.75`` for the input --> feature layer and ``0.5`` for feature --> output.

.. important::

   It is highly recommended that the lower threshold bound is kept to 0 (default) and the output layer higher threshold is kept to 0.5 (default).
   This is based on the learning rule as explored in the :doc:`training overview <train_overview>`.

Firing rates
^^^^^^^^^^^^^^^^^
The firing rates are used to adjust the spiking threshold values during weight updates. Increasing the firing rate has a greater effect
on modifying the firing thresholds. Like the thresholds, these values are set in a linspace range from low to high:

.. code-block:: bash

   pixi run train --fire_l_feat 0.4 --fire_h_feat 0.6 --fire_l_out 0.5 --fire_h_out 0.5

Saving trained models
-----------------
Models are automatically saved at the end of training all the layers. The model name uses the following convention:

.. code-block:: python

   model_name = "<reference>_<query>_IN<NUM_INPUT_NEURONS>_FN<NUM_FEATURE_NEURONS>_DB<NUM_REFERENCE_PLACES>.pth"

These unique model names allow you to train multiple networks on the same dataset with different network architectures to compare performance.

.. note::

   All models can be found in the ``./lens/models/`` subfolder.

When running the evaluation network, setting up the same dataset and network configuration will load the correct corresponding model.
