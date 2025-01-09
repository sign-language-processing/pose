import copy
from typing import List
import pytest
from pose_format import Pose
from pose_format.utils.generic import get_standard_components_for_known_format, fake_pose
@pytest.fixture
def fake_poses(request) -> List[Pose]:
    # Access the parameter passed to the fixture
    known_format = request.param
    components = get_standard_components_for_known_format(known_format)
    return copy.deepcopy([fake_pose(i * 10 + 10, components=components) for i in range(3)])
