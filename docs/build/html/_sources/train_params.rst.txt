Training Parameters
===================

Each of the following parameters can be passed to the LENS command-line interface to configure the training process.

.. list-table:: LENS Training Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Argument
     - Type
     - Description

   * - ``--dataset``
     - ``str``
     - Name of the dataset to use for training and/or inference.

   * - ``--camera``
     - ``str``
     - Type of camera used to capture the data (e.g., ``davis128``).

   * - ``--data_name``
     - ``str``
     - Identifier for the dataset instance, used for naming outputs.

   * - ``--reference``
     - ``str``
     - Directory name of the reference dataset.

   * - ``--query``
     - ``str``
     - Directory name of the query dataset.

   * - ``--data_dir``
     - ``str``
     - Path to the base directory containing the data folders.

   * - ``--reference_places``
     - ``int``
     - Number of locations (images) to use in the reference set.

   * - ``--query_places``
     - ``int``
     - Number of locations (images) to use in the query set.

   * - ``--feature_multiplier``
     - ``float``
     - Multiplier for expanding the feature layer size.

   * - ``--filter``
     - ``int``
     - Skip every nth image during dataset iteration (used for downsampling).

   * - ``--epoch_feat``
     - ``int``
     - Number of training epochs for the feature layer.

   * - ``--epoch_out``
     - ``int``
     - Number of training epochs for the output layer.

   * - ``--thr_l_feat``
     - ``float``
     - Lower threshold for membrane potential in the feature layer.

   * - ``--thr_h_feat``
     - ``float``
     - Upper threshold for membrane potential in the feature layer.

   * - ``--fire_l_feat``
     - ``float``
     - Lower bound for firing probability in the feature layer.

   * - ``--fire_h_feat``
     - ``float``
     - Upper bound for firing probability in the feature layer.

   * - ``--ip_rate_feat``
     - ``float``
     - Intrinsic plasticity learning rate for the feature layer.

   * - ``--stdp_rate_feat``
     - ``float``
     - STDP learning rate for the feature layer.

   * - ``--thr_l_out``
     - ``float``
     - Lower threshold for membrane potential in the output layer.

   * - ``--thr_h_out``
     - ``float``
     - Upper threshold for membrane potential in the output layer.

   * - ``--fire_l_out``
     - ``float``
     - Lower bound for firing probability in the output layer.

   * - ``--fire_h_out``
     - ``float``
     - Upper bound for firing probability in the output layer.

   * - ``--ip_rate_out``
     - ``float``
     - Intrinsic plasticity learning rate for the output layer.

   * - ``--stdp_rate_out``
     - ``float``
     - STDP learning rate for the output layer.

   * - ``--f_exc``
     - ``float``
     - Probability of excitatory connections in the feature layer.

   * - ``--f_inh``
     - ``float``
     - Probability of inhibitory connections in the feature layer.

   * - ``--o_exc``
     - ``float``
     - Probability of excitatory connections in the output layer.

   * - ``--o_inh``
     - ``float``
     - Probability of inhibitory connections in the output layer.

   * - ``--dims``
     - ``int``
     - Size to resize input images to before feeding into the network.

   * - ``--roi_dim``
     - ``int``
     - Size of the region of interest (ROI) that the network processes.

   * - ``--GT_tolerance``
     - ``int``
     - Ground truth tolerance for considering a match correct.

   * - ``--train_model``
     - ``flag``
     - Enables training mode. If not set, the system will run in inference or demo mode.