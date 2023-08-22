Structure
=============

The :ref:`pose_format` package stands as an exemplar in the domain of pose data management and analysis.
 Equipped with a multitude of functionalities, it is similar to a Swiss Army knife for everything related to pose data. 
 From reading and manipulation to visualization and testing, the package has a suite of tools.

Main Features
-------------

1. **Reading and Manipulating Pose Data**:
   
   * '.pose' files ensuring cross-compatibility between popular libraries such as NumPy, PyTorch, and TensorFlow.
   * The loaded data presents multiple manipulation options including:
     
     - Normalizing pose data.
     - Agumentation of data.
     - Interpolation of data.

2. **Visualization Capabilities**:

   - Methods to visualize raw and processed pose data using :class:`pose_format.pose_visualizer.PoseVisualizer` module.
   - Includes overlay functions for videos.

3. **Package Organization and Components**:

   * Structured with submodules and subpackages serving the purposes:
     - :ref:`pose_format_numpy` for NumPy interactions.
     - :ref:`pose_format_tensorflow`  for TensorFlow functionalities.
     - :ref:`pose_format_torch` for PyTorch-related tools.
     - :ref:`pose_format_third_party` for externals.
     - :ref:`_pose_format_utils` for utility tools. 

4. **Testing Suite**:

   - Ensuring the reliability of the package through robust testing tools and setups in the :ref:`pose_format_testing` subpackage.

Tests 
------

.. toctree::
   :maxdepth: 4

   tests
