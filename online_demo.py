# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import scipy.io
import pickle
import ipdb
from PIL import Image

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/P800853_11_10_22_run3-ezgifcom-video-cutter.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./P800853_11_10_22_run3-ezgifcom-video-cutter_Instrument.pkl",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--queries_path",
        default="./assets/PosDataMC_ARPS_Full.mat",
        help="path to points and timestamps to track"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    # segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    # segm_mask = torch.from_numpy(segm_mask)[None, None]

    # Load the tool segmentation masks
    # The format of segm_mask is a list of 2 element lists, where the first element 
    # is the frame number, and the second element is a 2D array representing the tool
    # segmentation mask on that frame
    with open(args.mask_path, 'rb') as pkl: 
        segm_mask = pickle.load(pkl)

    # queries_filename = os.path.basename(args.queries_path).split('.')[0]
    # queries = scipy.io.loadmat(args.queries_path)[queries_filename]
    # queries = filter(lambda query: query[0] != 0 and query[1] != 0, queries)
    # queries = [[float(index), row[0], row[1]] for index, row in enumerate(queries)][1:]

    # Load the .mat with the aiming beam data
    queries = scipy.io.loadmat(args.queries_path)['posMC']

    # We only want the first frame that each point appears on, so that we can track it ourselves
    queries = [[float(i), *queries[i][0][i]] for i in range(np.shape(queries)[0])]

    # Change queries so that each point only starts being tracked on the first frame
    # it isn't occluded by the tool
    initialized_queries = []
    while len(queries) > 0 and len(segm_mask) > 0:
        frame_number, mask = segm_mask.pop(0)
        i = 0

        while i < len(queries):
            x = int(queries[i][1])
            y = int(queries[i][2])
            RADIUS = 3
            area = np.transpose(mask[y - RADIUS: y + RADIUS])[x - RADIUS:x + RADIUS]
            if queries[i][0] <= frame_number:
                if (area > 0).any():
                    i = i + 1
                else:
                    queries[i][0] = frame_number
                    initialized_queries.append(queries.pop(i))
            else:
                i = i + 1

    queries = torch.tensor([*initialized_queries, *queries]).float()

    if torch.cuda.is_available():
        queries = queries.cuda()

    window_frames = []

    def _process_step(window_frames, queries, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            queries=queries[None],
            is_first_step=is_first_step,
            # segm_mask=segm_mask,
            # grid_size=grid_size,
            # grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    time_for_each_step = []
    scale = 1
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        width, height, _ = [int(dim * scale) for dim in np.shape(frame)]
        if i % model.step == 0 and i != 0:
            frame_start_time = time.time()
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                queries,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            frame_end_time = time.time()
            time_for_each_step.append(frame_end_time - frame_start_time)
            is_first_step = False
        window_frames.append(frame)
        print('{} frames processed'.format(i), end='\r')
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        queries,
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )

    print("Tracks are computed")
    print("Average time for each step: " + str(np.average(time_for_each_step)))
    print("Frames per step: " + str(model.step))
    print("Seconds of processing per second of video: " + str(np.average(time_for_each_step) * (30 / model.step)))

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=1)
    vis.visualize(video, pred_tracks, pred_visibility, queries, query_frame=args.grid_query_frame)
