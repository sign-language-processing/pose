.. _pose_format:

pose\_format package
====================

Here's a breakdown of the main subpackages:

- **NumPy package** e.g `pose_format.numpy`:
  
  This subpackage offers integration with the popular NumPy library, facilitating numeric computations with pose data.

  - `pose_format.numpy.pose_body` module and subpackages
    
    Provides utilities to handle body data in the context of pose with NumPy.

- **TensorFlow package** e.g `pose_format.tensorflow`:
  
  Aimed at users of the TensorFlow library, this subpackage offers seamless operations on pose data in TensorFlow contexts.

  - `pose_format.tensorflow.pose_body`:
    
    Main module for handling body data within TensorFlow.

  - `pose_format.tensorflow.pose_body_test`:
    
    Test module to ensure the accuracy and reliability of the pose body operations in TensorFlow.

  - `pose_format.tensorflow.pose_representation` and `pose_format.tensorflow.pose_representation_test`:
    
    These modules cater to the representation of poses in a TensorFlow environment. The test module ensures its robustness.

- **Testing package** e.g `pose_format.testing`:

  This subpackage provides a suite of tests to ensure the reliability of the entire `pose_format` package.

- **Third-Party package** e.g `pose_format.third_party`:

  Contains modules and utilities that integrate third-party tools or libraries with the pose format.

- **PyTorch package**:
  
  For PyTorch users, this subpackage provides modules to manage and manipulate pose data within a PyTorch context.

  - `pose_format.torch.pose_body`:
    
    Module dedicated to handling pose body data within PyTorch.

  - `pose_format.torch.pose_representation`:
    
    Aids in representing pose data within the PyTorch environment.

- **Utils package** e.g `pose_format.utils`:
  
  This subpackage consists of various utility modules to aid in generic operations related to pose data.


Subpackages
-----------

.. toctree::
   :maxdepth: 4

   pose_format.numpy
   pose_format.tensorflow
   pose_format.testing
   pose_format.third_party
   pose_format.torch
   pose_format.utils
    

Submodules
----------

pose\_format.pose module
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pose_format.pose
   :members:
   :undoc-members:
   :show-inheritance:

pose\_format.pose\_body module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pose_format.pose_body
   :members:
   :undoc-members:
   :show-inheritance:

pose\_format.pose\_header module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pose_format.pose_header
   :members:
   :undoc-members:
   :show-inheritance:

pose\_format.pose\_representation module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pose_format.pose_representation
   :members:
   :undoc-members:
   :show-inheritance:

pose\_format.pose\_test module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pose_format.pose_test
   :members:
   :undoc-members:
   :show-inheritance:

pose\_format.pose\_visualizer module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pose_format.pose_visualizer
   :members:
   :undoc-members:
   :show-inheritance:
