Using the optimizer
==================
Before proceeding, you will need to `sign up for a Weights & Biases account <https://wandb.ai/site/>`_.

Once you have an account, follow the `Quickstart instructions <https://docs.wandb.ai/quickstart/>`_ to connect your account to your local
work environment. 

.. important::

    Do not run ``pip install wandb``, this is included already in the ``pixi.toml`` environment.

Once you have logged in, you will not be prompted to login again.

Running the optimizer script
-----------------
To run the optimizer, we run the following in the command terminal:

.. code-block:: bash

    pixi run optimizer

Like the training and evaluation network, we can specify the dataset, camera, and reference/query data to optimize:

.. code-block:: bash

    pixi run optimizer --dataset example --camera davis128 --reference example-reference --query example-query

.. important::

    Please ensure you have set up your datasets correctly including the ground truth file. See :doc:`Training New Models <train_setup>` for
    more details.

Hyperparameters
--------------

There are several hyperparameters that can be tuned for your model. LENS allows users to set different hyperparameters for the **Input → Feature** and **Feature → Output** layers to give better flexibility when training models.

For a full list and description of supported training hyperparameters, see the :doc:`Hyperparameter list <train_params>`.

All hyperparameters are of type ``float`` with values in the range ``[0, 1]``, except for ``epoch_feat`` and ``epoch_out``, which are ``int`` values ≥ 1.

Good Practices
--------------

Whilst there are no hard and fast rules for setting hyperparameters, here are some general recommendations:

- Inhibitory connection probabilities are usually higher than excitatory ones (unless using full connectivity).
- For the output layer, keep the firing rate and threshold near 0.5. This helps with the delta-based learning rule.
- Use fewer training epochs for the Input → Feature layer to avoid overfitting the reference data.
- It is usually good to keep ``thr_l`` fixed at ``0``.

If you're unsure where to start, you can:

- Leave most values at their defaults and modify one at a time.
- Open an `issue on GitHub <https://github.com/AdamDHines/LENS/issues>`_ and we can help suggest tuning strategies.

Setting up Hyperparameter Sweeps
--------------

To automatically iterate over different hyperparameters using Weights & Biases (Wandb), edit the dictionary in the `optimizer.py` script at **line 64**.

Example:

.. code-block:: python

   parameters_dict = {    
       'fire_l_feat': {'values': [0.1, 0.2, 0.3]},
       'fire_h_feat': {'values': [0.6, 0.7, 0.8]},
       'thr_h_feat': {'values': [0.3, 0.4, 0.5]},
   }

You can also use ``np.linspace`` to generate evenly spaced ranges:

.. code-block:: python

   parameters_dict = {    
       'fire_l_feat': {'values': list(np.linspace(0.1, 0.49, 3))},
       'fire_h_feat': {'values': list(np.linspace(0.5, 1.0, 3))},
       'thr_h_feat': {'values': list(np.linspace(0.1, 0.5, 3))},
   }

.. warning:: 

    Wandb can struggle with large parameter spaces. To avoid overly long sweep runs, try tuning one layer at a time.

Wandb Sweep Setup
-----------------

Once you've defined ``parameters_dict``, configure Wandb to use those values by updating the `wandsearch()` function:

.. code-block:: python

   # Initialize w&b search
   def wandsearch(config=None):
       with wandb.init(config=config):
           config = wandb.config
           args.fire_l_feat = config.fire_l_feat
           args.fire_h_feat = config.fire_h_feat
           args.thr_h_feat = config.thr_h_feat

Wandb will automatically assign new values to these arguments at each training iteration and log the results to your online dashboard.

Sweep Method Configuration
--------------------------

By default, Wandb uses random sampling:

.. code-block:: python

   sweep_config = {'method': 'random'}

You can switch to grid search like so:

.. code-block:: python

   sweep_config = {'method': 'grid'}

You can also give your sweep a unique project name to distinguish different experiments:

.. code-block:: python

   sweep_id = wandb.sweep(sweep_config, project="random-sweep-001")

Model Evaluation
--------------

With a provided GT (ground truth) file, LENS performs a Recall@N evaluation for ``N = [1, 5, 10, 15, 20, 25]``. These values are returned as an array, and we compute the **area under the curve (AUC)** using ``np.trapz`` to produce a single performance score.

Wandb will automatically log:

- The best model performance
- The hyperparameter configuration that achieved it

Custom Metrics (e.g., Recall@1)
-------------------------------

You can override the default metric to focus on a specific value, such as Recall@1:

.. code-block:: python

   metric = {'name': 'R1', 'goal': 'maximize'}
   wandb.log({"R1": R_all[0]})
   print("R1: ", R_all[0])