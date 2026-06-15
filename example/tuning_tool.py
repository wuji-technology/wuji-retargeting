"""Parameter Tuning Visualization Tool (Internal).

Interactive MuJoCo viewer for hand retargeting parameter tuning.
Displays three skeleton layers simultaneously:
- Orange: Raw MediaPipe input keypoints
- Cyan: Scaled target keypoints (after segment_scaling)
- White: Robot FK keypoints (actual robot joint positions)

Edit the retarget config YAML while the viewer is running to see
parameter changes take effect in real-time.

Usage:
    # Replay recording
    mjpython tuning_tool.py --play data/avp1.pkl --hand left

    # Wuji Glove live (via wuji_sdk)
    mjpython tuning_tool.py --wuji-glove --hand left
    mjpython tuning_tool.py --wuji-glove --hand right --device-name glove

    # MP4 video input
    mjpython tuning_tool.py --video data/right.mp4 --hand right

    # RealSense camera
    mjpython tuning_tool.py --realsense --hand right

    # ZED camera
    mjpython tuning_tool.py --zed --hand right
"""

import argparse
import logging
import pickle
import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting.viz import TuningViewer
from utils.config_paths import resolve_mjcf_path


LOGGER = logging.getLogger(__name__)


def run_recording_mode(args):
    """Run tuning tool with recorded data."""
    config_path = Path(__file__).parent / args.config
    viewer = TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(config_path),
        viz_config_path=str(Path(__file__).parent / args.viz_config) if args.viz_config else None,
        mjcf_path=resolve_mjcf_path(config_path),
    )

    pkl_path = Path(__file__).parent / args.play
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} frames from {pkl_path.name}")
    viewer.play_recording(data, fps=args.fps)


def _run_live_mode(args, input_device, mode_name):
    """Shared live mode loop for real-time input devices."""
    config_path = Path(__file__).parent / args.config

    viewer = TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(config_path),
        viz_config_path=str(Path(__file__).parent / args.viz_config) if args.viz_config else None,
        mjcf_path=resolve_mjcf_path(config_path),
    )

    hand_key = f"{args.hand}_fingers"
    running = True

    def signal_handler(_sig, _frame):
        nonlocal running
        running = False

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    print(f"Tuning Viewer - {mode_name}")
    print(f"  Config: {args.config}")
    print(f"  Hand: {args.hand}")
    print(f"  Edit config file to tune parameters (auto-reload)")
    print(f"{'=' * 50}")

    for i in range(viewer.model.nu):
        if viewer.model.actuator_ctrllimited[i]:
            ctrl_range = viewer.model.actuator_ctrlrange[i]
            viewer.data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            viewer.data.ctrl[i] = 0.0
    for _ in range(100):
        mujoco.mj_step(viewer.model, viewer.data)

    last_result = None

    with mujoco.viewer.launch_passive(viewer.model, viewer.data) as mj_viewer:
        viewer._set_camera(mj_viewer)

        while mj_viewer.is_running() and running:
            changed, new_config = viewer.config_watcher.check()
            if changed:
                viewer._reload_retargeter(new_config, [])
                viewer.retargeter.reset()

            viewer._check_highlight_timeout()

            try:
                fingers_data = input_device.get_fingers_data()
                raw_kp = fingers_data[hand_key]

                if raw_kp is not None and not np.allclose(raw_kp, 0):
                    last_result = viewer._process_frame(raw_kp)
                    # Remap URDF/Pinocchio qpos order -> MJCF order (same as the
                    # viewer's recording path); identity when orders already match.
                    viewer.data.qpos[:] = last_result["qpos"][viewer._qpos_perm]
                    mujoco.mj_forward(viewer.model, viewer.data)
            except Exception:
                LOGGER.exception("Failed to process a live tuning frame")

            if last_result is not None:
                with mj_viewer.lock():
                    viewer.drawer.draw(
                        mj_viewer.user_scn,
                        mediapipe_kp=last_result["mediapipe_kp"],
                        scaled_kp=last_result["scaled_kp"],
                        pinch_alphas=last_result.get("pinch_alphas"),
                    )

            mj_viewer.sync()
            time.sleep(0.01)

    signal.signal(signal.SIGINT, old_handler)

    if hasattr(input_device, "cleanup"):
        input_device.cleanup()

    print("\nViewer closed")


def run_wuji_glove_mode(args):
    """Run tuning tool with Wuji Glove (live mode via wuji_sdk)."""
    try:
        from input_devices.wuji_glove_device import WujiGloveDevice
    except ImportError:
        print("wuji_sdk is not installed.")
        print("Please install wuji_sdk to use --wuji-glove.")
        sys.exit(1)

    device = WujiGloveDevice(
        hand_side=args.hand,
        device_name=args.device_name,
        sn=args.glove_sn,
    )
    _run_live_mode(args, device, f"Wuji Glove Live (device={args.device_name})")


def run_video_mode(args):
    """Run tuning tool with video input."""
    from input_devices.video_mediapipe import VideoMediaPipe

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    video_device = VideoMediaPipe(
        video_path=args.video,
        hand_side=args.hand,
        playback_speed=1.0,
        loop=False,
        video_config=video_config,
        show_video=False,
    )

    # Extract all frames, then play as recording
    hand_key = f"{args.hand}_fingers"
    other_key = "left_fingers" if args.hand == "right" else "right_fingers"
    empty = np.zeros((21, 3), dtype=np.float32)

    data = []
    for kp in video_device.get_landmarks():
        frame = {
            hand_key: kp.copy() if kp is not None else empty.copy(),
            other_key: empty.copy(),
        }
        data.append(frame)

    if hasattr(video_device, "cleanup"):
        video_device.cleanup()

    if not data:
        print("Error: No frames extracted from video")
        sys.exit(1)

    print(f"Extracted {len(data)} frames")

    viewer = TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(config_path),
        viz_config_path=str(Path(__file__).parent / args.viz_config) if args.viz_config else None,
        mjcf_path=resolve_mjcf_path(config_path),
    )
    viewer.play_recording(data, fps=args.fps)


def run_realsense_mode(args):
    """Run tuning tool with RealSense camera."""
    from input_devices.realsense_mediapipe import RealsenseMediaPipe

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    device = RealsenseMediaPipe(
        hand_side=args.hand,
        video_config=video_config,
        show_video=args.show_video,
    )
    _run_live_mode(args, device, "RealSense Live")


def run_zed_mode(args):
    """Run tuning tool with ZED camera."""
    from input_devices.zed_mediapipe import ZedMediaPipe

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    device = ZedMediaPipe(
        hand_side=args.hand,
        video_config=video_config,
        show_video=args.show_video,
    )
    _run_live_mode(args, device, "ZED Live")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Tuning Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Skeleton Colors:
  Orange  = Raw MediaPipe input keypoints
  Cyan    = Scaled target keypoints (after segment_scaling)
  White   = Robot FK keypoints (actual robot positions)

Examples:
  mjpython tuning_tool.py --play data/avp1.pkl --hand left
  mjpython tuning_tool.py --wuji-glove --hand right
  mjpython tuning_tool.py --video data/right.mp4 --hand right
  mjpython tuning_tool.py --realsense --hand right
  mjpython tuning_tool.py --zed --hand right
        """,
    )

    parser.add_argument("--config", type=str, default="config/adaptive_analytical_avp.yaml",
                        help="Path to retarget YAML config")
    parser.add_argument("--viz-config", type=str, default=None,
                        help="Path to visualization config")
    parser.add_argument("--hand", type=str, default="left", choices=["left", "right"],
                        help="Hand side (default: left)")
    parser.add_argument("--play", type=str, default=None, metavar="FILE",
                        help="Play recording file (.pkl)")
    parser.add_argument("--video", type=str, default=None, metavar="FILE",
                        help="MP4 video file with MediaPipe hand detection")
    parser.add_argument("--realsense", action="store_true",
                        help="Use RealSense camera input")
    parser.add_argument("--zed", action="store_true",
                        help="Use ZED camera input")
    parser.add_argument("--wuji-glove", action="store_true",
                        help="Use Wuji Glove input via wuji_sdk")
    parser.add_argument("--device-name", type=str, default="glove",
                        help="SDK device name for Wuji Glove (default: glove)")
    parser.add_argument("--glove-sn", type=str, default="",
                        help="Wuji Glove serial number (required when multiple Wuji devices online)")
    parser.add_argument("--show-video", action="store_true",
                        help="Show 2D video overlay (camera mode)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Playback FPS (default: 30)")

    args = parser.parse_args()

    # Auto-switch config
    if args.config == "config/adaptive_analytical_avp.yaml":
        if args.wuji_glove:
            args.config = f"config/adaptive_analytical_wuji_glove_{args.hand}.yaml"
        elif args.video or args.realsense or args.zed:
            args.config = "config/adaptive_analytical_video.yaml"

    # Default viz config
    if args.viz_config is None:
        default_viz = Path(__file__).parent / "config" / "tuning_viz.yaml"
        if default_viz.exists():
            args.viz_config = "config/tuning_viz.yaml"

    # Determine mode
    if args.wuji_glove:
        run_wuji_glove_mode(args)
    elif args.zed:
        run_zed_mode(args)
    elif args.realsense:
        run_realsense_mode(args)
    elif args.video:
        run_video_mode(args)
    elif args.play:
        run_recording_mode(args)
    else:
        args.play = "data/avp1.pkl"
        run_recording_mode(args)


if __name__ == "__main__":
    main()
