.. _tests: 

tests package
=============

The `tests` package is a part of the `pose_format` package,
ensuring the robustnes of the package. It also has designed test datasets and submodules for testing.

- **Test Data**: This section has the datasets used for testing the package.

- **tests Submodules**: These are  testing modules, each is looking into specific functionalities of the `pose_format` package.

  - ``tests.hand_normalization_test``: Focuses on validating the accuracy and correctness of 3D hand pose normalization techniques.
  
  - ``tests.optical_flow_test``: For tesing the implementation of optical flow algorithms.
  
  - ``tests.pose_tf_graph_mode_test``: Set of tests dedicated to test the behavior and output of TensorFlow operations when run in graph mode.
  
  - ``tests.pose_test``: A broad-spectrum testing module, probing a plethora of pose-related functionalities. It delves into standard pose operations and further evaluates their interplay with platforms like Numpy and TensorFlow.

.. toctree::
   :maxdepth: 3

   test_data
   tests_subs
