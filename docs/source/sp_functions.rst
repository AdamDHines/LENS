Speck Deployment Parameters
===========================

The following parameters are relevant when running LENS in **event-driven mode** on the Speck2fDevKit. These include options for data collection, on-chip deployment, simulation, and saving inputs.

.. list-table:: LENS Speck2fDevKit Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Argument
     - Type
     - Description

   * - ``--event_driven``
     - ``flag``
     - Enables event-driven mode using the on-chip LENS model deployed to the Speck2fDevKit. Required to trigger live inferencing from event streams.

   * - ``--simulated_speck``
     - ``flag``
     - Runs an offline, time-based simulation of the Speck inferencing model using saved spike data. Useful for debugging or replicating deployment behavior without hardware.

   * - ``--collect_data``
     - ``flag``
     - Activates data collection mode from the Speck2fDevKit. Captured spikes are converted into images and saved for training new models.

   * - ``--save_input``
     - ``flag``
     - If set, saves the raw input spike events as NumPy arrays for later analysis or use in simulated inference.

   * - ``--headless``
     - ``flag``
     - Runs the system without a GUI preview. This is essential for deployment on headless devices or remote operation.

   * - ``--dataset``
     - ``str``
     - Top-level dataset folder under ``lens/dataset/`` where captured or processed data will be saved and referenced.

   * - ``--camera``
     - ``str``
     - Subfolder indicating the camera device or identifier (e.g., ``speck``, ``speckkit001``).

   * - ``--reference``
     - ``str``
     - Name of the subfolder containing reference images or pre-captured spikes used to initialize the model.

   * - ``--reference_places``
     - ``int``
     - Number of reference places (images) to load into memory for place recognition.

   * - ``--dims``
     - ``list[int, int]``
     - Size to resize input images to before feeding into the network.

   * - ``--roi_dim``
     - ``list[int, int]``
     - Size of the input images for the network.

   * - ``--timebin``
     - ``int``
     - Spike integration window in milliseconds (e.g., ``1000`` for 1 second). Defines the time frame for collecting input before inference.