Examples
========

Pose Format Conversion with Sirens
-----------------------------------

The ``pose_to_siren_to_pose.py`` script shows us how to change a 3D pose using something called a Siren neural network.

.. note::

   The function follows these steps:
   
   1. Fills missing data in the body with zeros.
   2. Normalizes the pose distribution.
   3. Converts the pose using the Siren neural network.
   4. Constructs the new Pose with the predicted data.
   5. Unnormalizes the pose distribution.

   Also:  Before you start, ensure you have a `.pose` file/ path. This is the standardized format that stores the 3D pose information. If you don't have one, you might either need to obtain it from a relevant dataset or convert your existing pose data into this format.

Step-by-step Guide:
~~~~~~~~~~~~~~~~~~~

1. **Preparation**

    Begin by importing the necessary modules:

    .. code-block:: python

       import numpy as np
       from numpy import ma
       import pose_format.utils.siren as siren
       from pose_format import Pose
       from pose_format.numpy import NumPyPoseBody
       from pose_format.pose_visualizer import PoseVisualizer


2. **Define the Conversion Function**

    The function `pose_to_siren_to_pose` is used to perform the conversion. An you find the overview of the whole function :ref:`overview`


    .. code-block:: python

        def pose_to_siren_to_pose(p: Pose, fps=None) -> Pose:
        """Converts a given Pose object to its Siren representation and back to Pose."""

            # Fills missing values with 0's 
            p.body.zero_filled()

            # Noralizes
            mu, std = p.normalize_distribution()

            # Use siren net 
            net = siren.get_pose_siren(p, total_steps=3000, steps_til_summary=100, learning_rate=1e-4, cuda=True)

            new_fps = fps if fps is not None else p.body.fps
            coords = siren.PoseDataset.get_coords(time=len(p.body.data) / p.body.fps, fps=new_fps)

            # Get predicitons of new Pose data 
            pred = net(coords).cpu().numpy()

            # Construct new Body out of predcitions 
            pose_body = NumPyPoseBody(fps=new_fps, data=ma.array(pred), confidence=np.ones(shape=tuple(pred.shape[:3])))
            p = Pose(header=p.header, body=pose_body)

            # Revert normalization and give back the pose instance
            p.unnormalize_distribution(mu, std)
            return p


    The function does the following operations:

    - Fills missing data in the pose body with zeros.
    - Normalizes the pose distribution.
    - Uses the Siren neural network to transform the pose.
    - Constructs a new Pose with the predicted data.
    - Reverts the normalization on the pose distribution.

3. **Usage**

    After defining the function, you can use it in your main script:

    .. code-block:: python

       if __name__ == "__main__":
        pose_path = "/home/nlp/amit/PhD/PoseFormat/sample-data/1.pose"  # your own file path to a `.pose` file 

        buffer = open(pose_path, "rb").read()
        p = Pose.read(buffer)
        print("Poses loaded")

        p = pose_to_siren_to_pose(p)

        info = p.header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder")
        )
        p.normalize(info, scale_factor=300)
        p.focus()

        v = PoseVisualizer(p)
        v.save_video("reconstructed.mp4", v.draw(max_frames=3000))


    The main script performs these tasks:

    - Reads the pose data from a file.
    - Applies the ``pose_to_siren_to_pose`` function to the read pose.
    - Normalizes and focuses the pose.
    - Visualizes the converted pose using the ``PoseVisualizer``.

4. **Execution**

    To run the script:

    .. code-block:: bash

       $ python pose_format_converter.py



The ``pose_format`` combined with Siren neural networks is great to transform and work with 3D pose data.
By understanding and using the functions and methods provided in this script, you will be able to understand better how to manipulate and visualize 3D poses to suit your own requirements.


.. _overview:

Overview of ``pose_to_siren_to_pose.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/pose_to_siren_to_pose.py
   :language: python
   :linenos: