# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
eval_fixed_horizon.py -- compatibility entry point; the evaluation lives in ``eval.py``.

The Eq. 15 options first added here (``--fixed-horizon``, ``--beta``, ``--min-horizon``,
``--suggest-beta``, per-step sigma / H_t in the results) are now part of ``eval.py``, together with
multi-camera checkpoints. This file only forwards, so existing commands keep working and give the
same results:

    python eval_fixed_horizon.py --checkpoint ... --fixed-horizon 4    # same as: python eval.py ...
    python eval_fixed_horizon.py --self-test                           # runs eval.py's self-test
"""

from __future__ import annotations

try:  # package import
    from .eval import (
        CAMERA_OBS_KEYS,
        LIBERO_DEPENDENCIES,
        PROPRIO_FROM_OBS,
        SUITES,
        DemoMilestones,
        EpisodeContext,
        FramePreprocessor,
        beta_for,
        build_arg_parser,
        evaluate,
        import_libero,
        load_init_states,
        main,
        make_env,
        make_suite,
        proprio_from_obs,
        quat2axisangle,
        run_episode,
        save_video,
        suggest_beta,
    )
except ImportError:  # flat import (files side by side, `python eval_fixed_horizon.py`)
    from eval import (
        CAMERA_OBS_KEYS,
        LIBERO_DEPENDENCIES,
        PROPRIO_FROM_OBS,
        SUITES,
        DemoMilestones,
        EpisodeContext,
        FramePreprocessor,
        beta_for,
        build_arg_parser,
        evaluate,
        import_libero,
        load_init_states,
        main,
        make_env,
        make_suite,
        proprio_from_obs,
        quat2axisangle,
        run_episode,
        save_video,
        suggest_beta,
    )

__all__ = [
    "CAMERA_OBS_KEYS",
    "LIBERO_DEPENDENCIES",
    "PROPRIO_FROM_OBS",
    "SUITES",
    "DemoMilestones",
    "EpisodeContext",
    "FramePreprocessor",
    "beta_for",
    "build_arg_parser",
    "evaluate",
    "import_libero",
    "load_init_states",
    "main",
    "make_env",
    "make_suite",
    "proprio_from_obs",
    "quat2axisangle",
    "run_episode",
    "save_video",
    "suggest_beta",
]

if __name__ == "__main__":
    main()
