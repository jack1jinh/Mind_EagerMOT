# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""EagerMOT backbone"""

from typing import List
from utils import track_function
from config.params import KITTI_BEST_PARAMS
from config.local_variables import KITTI_WORK_DIR
import engine.utils as input_utils
import dataset.kitti.mot_kitti as mot_kitti


def run_on_kitti():
    """
    EagerMOT algorithm backbone
    """

    mot_dataset = mot_kitti.MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
                                            det_source=input_utils.POINTGNN_T3,
                                            seg_source=input_utils.TRACKING_BEST)

    # if want to run on specific sequences only, add their str names here
    target_sequences: List[str] = []

    # if want to exclude specific sequences, add their str names here
    sequences_to_exclude: List[str] = []

    track_function.perform_tracking_with_params(mot_dataset, KITTI_BEST_PARAMS, target_sequences, sequences_to_exclude)


if __name__ == "__main__":
    run_on_kitti()
