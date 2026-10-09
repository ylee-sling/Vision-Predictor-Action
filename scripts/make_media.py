# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
make_media.py -- README GIFs and a video archive from ``eval.py --save-video`` runs.

``eval.py --save-video`` writes one MP4 per episode, named ``task<TT>_trial<RR>_<success|failure>.mp4``
(20 control steps per second; two-camera runs show the third-person and wrist views side by side).
This script

    * converts chosen episodes of each run to small looping GIFs for the README
      (``docs/media/task<TT>_trial<RR>_<label>.gif``), sped up and frame-subsampled;
    * zips every video of the given runs into one archive for a GitHub release.

Usage:
    python scripts/make_media.py \
        --run 1cam=eval_results/videos/1cam/videos --run 2cam=eval_results/videos/2cam/videos \
        --episodes 5:1 7:0 8:2 --out docs/media --zip ~/videos_libero10_pilot.zip

Needs imageio with imageio-ffmpeg (to read MP4) and Pillow (to write GIF).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np

NAME = re.compile(r"task(\d+)_trial(\d+)_(success|failure)\.mp4$")


def find_episode(folder: str, task: int, trial: int) -> Tuple[str, str]:
    """Path and outcome ("success"/"failure") of one episode video in ``folder``."""
    hits = glob.glob(os.path.join(folder, f"task{task:02d}_trial{trial:02d}_*.mp4"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one video for task {task} trial {trial} in {folder}, found {hits}")
    match = NAME.search(os.path.basename(hits[0]))
    assert match is not None
    return hits[0], match.group(3)


def read_frames(path: str) -> List[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def write_gif(frames: Sequence[np.ndarray], path: str, every: int, fps: float, scale: int) -> None:
    """Every ``every``-th frame, enlarged ``scale`` times (nearest neighbour), played at ``fps``, looping."""
    from PIL import Image

    picked = list(frames[::every])
    if not picked:
        raise ValueError(f"no frames for {path}")
    images = []
    for frame in picked:
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000 / fps)), loop=0,
                   optimize=True)


def parse_runs(items: Sequence[str]) -> Dict[str, str]:
    runs: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--run expects LABEL=FOLDER, got {item!r}")
        label, folder = item.split("=", 1)
        if not os.path.isdir(folder):
            raise SystemExit(f"not a folder: {folder}")
        runs[label] = folder
    return runs


def _self_test() -> None:
    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        frames = [np.full((8, 16, 3), i * 10, dtype=np.uint8) for i in range(20)]
        out = os.path.join(tmp, "x.gif")
        write_gif(frames, out, every=4, fps=10, scale=2)
        with Image.open(out) as gif:
            assert gif.size == (32, 16) and gif.n_frames == 5
        open(os.path.join(tmp, "task05_trial01_success.mp4"), "w").close()
        assert find_episode(tmp, 5, 1) == (os.path.join(tmp, "task05_trial01_success.mp4"), "success")
    print("make_media.py self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=FOLDER",
                        help="a folder of eval.py videos and the label used in GIF names (repeat)")
    parser.add_argument("--episodes", nargs="*", default=[], metavar="TASK:TRIAL",
                        help="episodes to convert to GIF for every run, e.g. 5:1 7:0 8:2")
    parser.add_argument("--out", default="docs/media", help="GIF folder")
    parser.add_argument("--zip", help="also write all videos of the runs to this zip archive")
    parser.add_argument("--every", type=int, default=3, help="keep every n-th frame (default 3)")
    parser.add_argument("--fps", type=float, default=13.3, help="GIF playback rate (default 13.3: 2x real time)")
    parser.add_argument("--scale", type=int, default=2, help="enlarge frames n times (default 2)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    runs = parse_runs(args.run)
    if not runs:
        parser.error("give at least one --run LABEL=FOLDER")
    os.makedirs(args.out, exist_ok=True)
    for item in args.episodes:
        task, trial = (int(v) for v in item.split(":"))
        for label, folder in runs.items():
            src, outcome = find_episode(folder, task, trial)
            dst = os.path.join(args.out, f"task{task:02d}_trial{trial:02d}_{label}.gif")
            write_gif(read_frames(src), dst, args.every, args.fps, args.scale)
            print(f"{dst}  ({outcome}, {os.path.getsize(dst) / 1e6:.2f} MB)")
    if args.zip:
        count = 0
        with zipfile.ZipFile(os.path.expanduser(args.zip), "w", compression=zipfile.ZIP_STORED) as zf:
            for label, folder in runs.items():
                for path in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
                    zf.write(path, arcname=os.path.join(label, os.path.basename(path)))
                    count += 1
        print(f"{args.zip}: {count} videos, {os.path.getsize(os.path.expanduser(args.zip)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
