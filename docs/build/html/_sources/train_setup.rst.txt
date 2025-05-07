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

For datasets where there is a 1:1 correspondance between queries and reference, the ground truth can simply be:

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

If we had a different dataset, for example we collected two different event streams from an outdoor environment (OutDoorEv) on a DAVIS346 with 500 images each,
we would modify the arguments as:

.. code-block:: bash

   pixi run train --dataset OutDoorEv --camera DAVIS346 --reference stream001 --reference_places 500

.. hint::

   However the data is stored in ``./lens/dataset/`` is what you will call in the arguments

If your dataset is in a location other than ``./lens/dataset/``, you can change the root dataset directory path:

.. code-block:: bash

   pixi run train --data_dir <your_dataset_location>/

Training Parameters
-----------------
There are a plethora of training parameters that can be tuned for different datasets. In general, the default hyperparameters have been found to generalize
well to multiple different datasets from various event cameras.

.. note::

   Please see :doc:`Training Parameters <train_params>` for a full list of training hyperparameters.

We can modify the number of neurons in our feature layer which will affect the spatial representation of information of input images. Increasing the number of
neurons generally increases performance, to a point, whilst decreasing the number of neurons decreases performance.

The ``--feature_multiplier`` argument controls how many feature neurons there are relative to the number of pixel encoding input neurons.

.. code-block:: bash

   pixi run train --feature_multiplier 4.0 # for 4x the number of neurons relative to input
   pixi run train --feature_multiplier 0.5 # for 1/2x the number of neurons relative to input