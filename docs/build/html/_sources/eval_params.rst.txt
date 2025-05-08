Evaluation Parameters
=====================

The following parameters are used during evaluation and inference of a trained LENS model. These options configure the dataset, reference/query pairings, and optional visual and metric-based evaluation tools.

.. list-table:: LENS Evaluation Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Argument
     - Type
     - Description

   * - ``--dataset``
     - ``str``
     - Top-level folder name under ``lens/dataset/`` that contains the dataset being used for reference and/or query.

   * - ``--camera``
     - ``str``
     - Subfolder indicating the DVS camera used (e.g., ``davis128``, ``speck``).

   * - ``--reference``
     - ``str``
     - Name of the reference experiment directory inside the dataset (used to load the trained model and evaluate against).

   * - ``--reference_places``
     - ``int``
     - Number of reference images to include during evaluation.

   * - ``--query``
     - ``str``
     - Name of the query experiment directory to be evaluated against the reference.

   * - ``--query_places``
     - ``int``
     - Number of query images to use for evaluation.

   * - ``--dims``
     - ``int``
     - Image size (height = width) expected by the trained model. Must match the resolution used during training.

   * - ``--sim_mat``
     - ``flag``
     - Generates a similarity matrix between reference and query images, useful for visual inspection of performance.

   * - ``--matching``
     - ``flag``
     - Enables evaluation against a ground truth `.npy` file with binary match labels.

   * - ``--GT_tolerance``
     - ``int``
     - Match is considered correct if it falls within this many indices of the true match (i.e., recall@tolerance).

   * - ``--PR_curve``
     - ``flag``
     - Outputs a `.json` file containing precision-recall curve data and generates a corresponding plot.

   * - ``--sad``
     - ``flag``
     - Runs SAD (Sum of Absolute Differences) matching instead of neural similarity, useful for baseline comparisons.

   * - ``--nocuda``
     - ``flag``
     - Disables GPU usage for inference, forcing all operations to run on CPU.