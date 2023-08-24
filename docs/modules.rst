Structure
=============

The :ref:`pose_format` aims to help users with the pose data management and pose analysis.
 The toolkit offers many functionalities, from reading and editing to visualizing and testing pose data. It provides a wide range of features for these tasks.

This section gives an brief overview of the main feature structure of the package and its functionalities.

Main Features
-------------

1. **Reading and Manipulating Pose Data**:
   
   * 	`.pose` files ensuring cross-compatibility between popular libraries such as NumPy, PyTorch, and TensorFlow.
   * The loaded data presents multiple manipulation options including:
     
     - Normalizing pose data.
     - Agumentation of data.
     - Interpolation of data.


2. **Visualization Capabilities**:

   - Methods to visualize raw and processed pose data using `pose_format.pose_visualizer.PoseVisualizer` module.
   - Includes overlay functions for videos.


3. **Package Organization and Components**:

   * Structured with submodules and subpackages serving the purposes:

      - :ref:`pose_format_numpy` for NumPy interactions.

      - :ref:`pose_format_tensorflow`  for TensorFlow functionalities.

      - :ref:`pose_format_torch` for PyTorch-related tools.

      - :ref:`pose_format_third_party` for externals.

      - :ref:`pose_format_utils` for utility tools. 

4. **Testing Suite**:

   - Tests for the reliability of the package and its setups/data can be found in :ref:`tests`.

Tests 
------

This section illustrates the content of the testing suite and the used data. 

.. toctree::
   :maxdepth: 4

   tests