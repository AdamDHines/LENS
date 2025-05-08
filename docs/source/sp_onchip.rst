On-device localization
===========================
We can run the online localization system, where place matching occurs on-device, by simply running:

.. code-block:: bash

    pixi run on-speck

This will spawn a ``samnagui`` GUI instance that will show you the input event stream and the power consumption for the 5 power tracks (IO, logic, RAM, VDDD, & VDDA) 
measured at 20 Hz. 

.. hint::
    
    It is possible to suppress this GUI by using the ``--headless`` arugment to reduce computational load during localization, however
    this prevents you from obtaining power measurements (described further down).

Online sequence matching
-----------------
Place predictions in the default setting occurs after 4-seconds, or after 4 different `places` have been traversed through. This uses the same
computational logic as the simulated events used on regular CPU hardware, where it will collect events over a 1-second timebin and use that
event stream as a **place**.

LENS will return the output event spikes over the 1-second timebin and concatenate 4 of these into a similarity matrix. A sequence matcher kernel
will then run and the place predictions will occur in real-time.

You can change the timewindow for collection by using the ``--timebin`` argument:

.. code-block:: bash

    pixi run on-speck --timebin 250

This will instead collect events for a place over 250 ms, instead of the default 1,000 ms.

Power measurements
-----------------
During online localization, power is continually monitored and recorded. In order to save the power measurements, the ``samnagui`` window
must be open during localization. Destroying the ``samnsagui`` window triggers the shutdown of SPECK and retrieval of stored power measurements.

All power measurements are saved to the output folder created in ``./lens/output/`` alongside similarity matrices and the ``.log`` file, under
``power_data.npy``.

The power measurements are stored in 5 separate ``numpy array`` in mW and are named according to the power track.

Saving input event streams
-----------------
The input event streams can also be saved as a ``numpy array`` if you wish to create images of them:

.. code-block:: bash

    pixi run on-speck --save_input

These will save to the same output folder under ``./lens/output/<output_name>/events/`` as a ``.npy`` file.

To easily turn these into images for later use, you can use the `event frame generator script <https://github.com/AdamDHines/LENS/blob/main/lens/tools/manual_eventframe_generator.py>`_
provided.