import time
from itertools import product
from typing import List, Iterable, Mapping
import os

import dataset.kitti.mot_kitti as mot_kitti
from dataset.nuscenes.dataset import MOTDatasetNuScenes
from utils import io
from config.params import TRAIN_SEQ, VAL_SEQ, TRACK_VAL_SEQ, build_params_dict, KITTI_BEST_PARAMS, NUSCENES_BEST_PARAMS, variant_name_from_params
from config.local_variables import KITTI_WORK_DIR, SPLIT, NUSCENES_WORK_DIR, MOUNT_PATH
import engine.utils as input_utils

def perform_tracking_full(dataset, params, target_sequences=[], sequences_to_exclude=[], print_debug_info=True):

    if len(target_sequences) == 0:
        # 获取训练集或者测试集的序列
        target_sequences = dataset.sequence_names(SPLIT)

    total_frame_count = 0
    total_time = 0
    total_time_tracking = 0
    total_time_fusion = 0
    total_time_reporting = 0

    for sequence_name in target_sequences:
        if len(sequences_to_exclude) > 0:
            if sequence_name in sequences_to_exclude:
                print(f'Skipped sequence {sequence_name}')
                continue

        print(f'Starting sequence: {sequence_name}') 
        start_time = time.time()
        sequence = dataset.get_sequence(SPLIT, sequence_name)  # 获取当前序列下的片段所有相关信息
        sequence.mot.set_track_manager_params(params)  # 跟踪算法的一些对应参数
        variant = variant_name_from_params(params)
        run_info = sequence.perform_tracking_for_eval(params)
        if "total_time_mot" not in run_info:
            continue

        total_time = time.time() - start_time
        if print_debug_info:
            print(f'Sequence {sequence_name} took {total_time:.2f} sec, {total_time / 60.0 :.2f} min')
            print(
                f'Matching took {run_info["total_time_matching"]:.2f} sec, {100 * run_info["total_time_matching"] / total_time:.2f}%')
            print(
                f'Creating took {run_info["total_time_creating"]:.2f} sec, {100 * run_info["total_time_creating"] / total_time:.2f}%')
            print(
                f'Fusion   took {run_info["total_time_fusion"]:.2f} sec, {100 * run_info["total_time_fusion"] / total_time:.2f}%')
            print(
                f'Tracking took {run_info["total_time_mot"]:.2f} sec, {100 * run_info["total_time_mot"] / total_time:.2f}%')

            print(
                f'{run_info["matched_tracks_first_total"]} 1st stage and {run_info["matched_tracks_second_total"]} 2nd stage matches')

        total_time += total_time
        total_time_fusion += run_info["total_time_fusion"]
        total_time_tracking += run_info["total_time_mot"]
        total_time_reporting += run_info["total_time_reporting"]
        total_frame_count += len(sequence.frame_names)

    if total_frame_count == 0:
        return variant, run_info

    dataset.save_all_mot_results(run_info["mot_3d_file"])

    if not print_debug_info:
        return variant, run_info

    # Overall variant stats
    # Timing
    print("\n")
    print(
        f'Fusion    {total_time_fusion: .2f} sec, {(100 * total_time_fusion / total_time):.2f}%')
    print(f'Tracking  {total_time_tracking: .2f} sec, {(100 * total_time_tracking / total_time):.2f}%')
    print(f'Reporting {total_time_reporting: .2f} sec, {(100 * total_time_reporting / total_time):.2f}%')
    print(
        f'Tracking-fusion framerate: {total_frame_count / (total_time_fusion + total_time_tracking):.2f} fps')
    print(f'Tracking-only framerate: {total_frame_count / total_time_tracking:.2f} fps')
    print(f'Total framerate: {total_frame_count / total_time:.2f} fps')
    print()

    # Fused instances stats
    total_instances = run_info['instances_both'] + run_info['instances_3d'] + run_info['instances_2d']
    if total_instances > 0:
        print(f"Total instances 3D and 2D: {run_info['instances_both']} " +
              f"-> {100.0 * run_info['instances_both'] / total_instances:.2f}%")
        print(f"Total instances 3D only  : {run_info['instances_3d']} " +
              f"-> {100.0 * run_info['instances_3d'] / total_instances:.2f}%")
        print(f"Total instances 2D only  : {run_info['instances_2d']} " +
              f"-> {100.0 * run_info['instances_2d'] / total_instances:.2f}%")
        print()

    # Matching stats
    print(f"matched_tracks_first_total {run_info['matched_tracks_first_total']}")
    print(f"unmatched_tracks_first_total {run_info['unmatched_tracks_first_total']}")

    print(f"matched_tracks_second_total {run_info['matched_tracks_second_total']}")
    print(f"unmatched_tracks_second_total {run_info['unmatched_tracks_second_total']}")
    print(f"unmatched_dets2d_second_total {run_info['unmatched_dets2d_second_total']}")

    first_matched_percentage = (run_info['matched_tracks_first_total'] /
                                (run_info['unmatched_tracks_first_total'] + run_info['unmatched_tracks_first_total']))
    print(f"percentage of all tracks matched in 1st stage {100.0 * first_matched_percentage:.2f}%")

    second_matched_percentage = (
        run_info['matched_tracks_second_total'] / run_info['unmatched_tracks_first_total'])
    print(f"percentage of leftover tracks matched in 2nd stage {100.0 * second_matched_percentage:.2f}%")

    second_matched_dets2d_second_percentage = (run_info['matched_tracks_second_total'] / (
        run_info['unmatched_dets2d_second_total'] + run_info['matched_tracks_second_total']))
    print(f"percentage dets 2D matched in 2nd stage {100.0 * second_matched_dets2d_second_percentage:.2f}%")

    final_unmatched_percentage = (run_info['unmatched_tracks_second_total'] / (
        run_info['matched_tracks_first_total'] + run_info['unmatched_tracks_first_total']))
    print(f"percentage tracks unmatched after both stages {100.0 * final_unmatched_percentage:.2f}%")

    print(f"\n3D MOT saved in {run_info['mot_3d_file']}", end="\n\n")
    return variant, run_info


def perform_tracking_with_params(dataset, params,
                                 target_sequences: Iterable[str] = [],
                                 sequences_to_exclude: Iterable[str] = []):
    start_time = time.time()
    variant, run_info = perform_tracking_full(dataset, params,
                                              target_sequences=target_sequences,
                                              sequences_to_exclude=sequences_to_exclude)
    print(f'Variant {variant} took {(time.time() - start_time) / 60.0:.2f} mins')
    return run_info
