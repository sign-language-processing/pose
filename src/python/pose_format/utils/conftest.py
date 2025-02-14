import copy
from typing import List, get_args
import pytest
from pose_format.pose import Pose
from pose_format.utils.generic import get_standard_components_for_known_format, fake_pose, KnownPoseFormat

@pytest.fixture
def fake_poses(request) -> List[Pose]:
    # Access the parameter passed to the fixture
    known_format = request.param
    count = getattr(request, "count", 3)  
    known_formats = get_args(KnownPoseFormat)
    if known_format in known_formats:

        components = get_standard_components_for_known_format(known_format)
        return copy.deepcopy([fake_pose(i * 10 + 10, components=components) for i in range(count)])
    else:
        # get openpose
        fake_poses_list = [fake_pose(i * 10 + 10) for i in range(count)]
        for i, pose in enumerate(fake_poses_list):
            for component in pose.header.components:
                component.name = f"unknown_component_{i}_formerly_{component.name}"
        return [pose.copy() for pose in fake_poses_list]
