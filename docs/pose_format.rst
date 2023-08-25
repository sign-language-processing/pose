.. _pose_format:

pose\_format package
====================

Here's a breakdown of the main subpackages:

- **NumPy package** e.g `pose_format.numpy`:
  
  This subpackage offers an integration with the popular NumPy library and helps with numerical computations on pose data.

  - `pose_format.numpy.pose_body` module and subpackages:
    
    Provides tools to handle body data in context of pose with NumPy.

- **TensorFlow package** e.g `pose_format.tensorflow`:
  
  Provides integration with TensorFlow library, therefore this subpackage offers operations on pose data in TensorFlow contexts.

  - `pose_format.tensorflow.pose_body`:
    
    Main module for handling body data within TensorFlow.

  - `pose_format.tensorflow.pose_body_test`:
    
    Test module to ensure reliability of pose body operations in TensorFlow.

  - `pose_format.tensorflow.pose_representation` and `pose_format.tensorflow.pose_representation_test`:
    
    These modules help the representation of poses in TensorFlow.

- **Testing package** e.g `pose_format.testing`:

  This subpackage provides test cases to ensure reliability of entire `pose_format` package.

  - **Third-Party package** e.g `pose_format.third_party`:

  Contains modules and utilities that integrate third-party tools or libraries with pose format

- **PyTorch package** e.g `pose_format.torch`:
  
  This subpackage provides modules to manage and manipulate pose data in a PyTorch context

  - `pose_format.torch.pose_body`:
    
    Module for handling pose body data within PyTorch.

  - `pose_format.torch.pose_representation`:
    
    Helps representing pose data in PyTorch.

- **Utils package** e.g `pose_format.utils`:
  
  This subpackage consists of various modules to help operations related to pose data.



.. toctree::
   :maxdepth: 9
   :caption: Subpackages:

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
