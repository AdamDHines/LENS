Deploying to SPECK
===========================
The simplest way to get started deploying to SPECK is to run a simulated event stream on-chip:

.. code-block:: bash

    pixi run sim-speck

This utilizes the same spike generator described in :doc:`Evaluation Overview - Time-based frame conversion <eval_overview>` but converts them to SPECK events before transferring them
to the I/O of the development kit.

This will output the results in exactly the same way as described in :doc:`Evaluation Setup - Results output & saving <eval_setup>`.

.. warning::

    Running simulations on SPECK is slow due to the time necessary to generate events and send them to the development kit. 

Just like the evaluation network, specify the dataset and other necessary parameters if not running the demonstration example:

.. code-block:: bash

    pixi run sim-speck --dataset OutdoorNav --camera davis128 --reference traverse001 --query traverse002