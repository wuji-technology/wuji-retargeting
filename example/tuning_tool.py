"""Parameter Tuning Visualization Tool.

Interactive MuJoCo viewer for hand retargeting parameter tuning.
Displays three skeleton layers simultaneously:
- Orange: Raw MediaPipe input keypoints
- Cyan: Scaled target keypoints (after segment_scaling)
- White: Robot FK keypoints (actual robot joint positions)

Edit the retarget config YAML while the viewer is running to see
parameter changes take effect in real-time.

Usage:
    # Replay recording (default)
    mjpython tuning_tool.py --play data/avp1.pkl --hand left

    # With custom retarget config
    mjpython tuning_tool.py --play data/avp1.pkl --config config/adaptive_analytical_avp.yaml

    # With custom viz config
    mjpython tuning_tool.py --play data/avp1.pkl --viz-config config/tuning_viz.yaml

    # MP4 video input
    mjpython tuning_tool.py --video data/right.mp4 --hand right

    # RealSense camera
    mjpython tuning_tool.py --realsense --hand right

    # ZED camera
    mjpython tuning_tool.py --zed --hand right
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting.viz import TuningViewer


def run_recording_mode(args):
    """Run tuning tool with recorded data."""
    viewer = TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(Path(__file__).parent / args.config),
        viz_config_path=str(Path(__file__).parent / args.viz_config) if args.viz_config else None,
    )

    # Load recording
    pkl_path = Path(__file__).parent / args.play
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} frames from {pkl_path.name}")
    viewer.play_recording(data, fps=args.fps)


def run_video_mode(args):
    """Run tuning tool with video input (processes all frames first, then plays)."""
    try:
        from input_devices.video_mediapipe import VideoMediaPipe
    except ImportError:
        print("Error: video mode requires mediapipe and opencv-python")
        print("Install with: pip install wuji-retargeting[video]")
        sys.exit(1)

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    # Create video device to extract frames
    video_device = VideoMediaPipe(
        video_path=args.video,
        hand_side=args.hand,
        playback_speed=1.0,
        loop=False,
        video_config=video_config,
        show_video=False,
    )

    # Extract all pre-processed landmarks directly (avoid time-based playback API)
    hand_key = f"{args.hand}_fingers"
    other_key = "left_fingers" if args.hand == "right" else "right_fingers"
    empty = np.zeros((21, 3), dtype=np.float32)

    data = []
    for kp in video_device._landmarks:
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
    )
    viewer.play_recording(data, fps=args.fps)


def run_realsense_mode(args):
    """Run tuning tool with RealSense camera (live mode)."""
    try:
        from input_devices.realsense_mediapipe import RealsenseMediaPipe
    except ImportError:
        print("Error: realsense mode requires mediapipe, opencv-python, and pyrealsense2")
        print("Install with: pip install wuji-retargeting[realsense]")
        sys.exit(1)

    import mujoco
    import mujoco.viewer
    import signal

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    # Create viewer
    viewer = TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(config_path),
        viz_config_path=str(Path(__file__).parent / args.viz_config) if args.viz_config else None,
    )

    # Create RealSense device
    rs_device = RealsenseMediaPipe(
        hand_side=args.hand,
        video_config=video_config,
        show_video=args.show_video,
    )

    hand_key = f"{args.hand}_fingers"
    running = True

    def signal_handler(_sig, _frame):
        nonlocal running
        running = False

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    print(f"Tuning Viewer - RealSense Live Mode")
    print(f"  Config: {args.config}")
    print(f"  Hand: {args.hand}")
    print(f"  Edit config file to tune parameters (auto-reload)")
    print(f"{'=' * 50}")

    # Initialize model
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
            # Check config
            changed, new_config = viewer.config_watcher.check()
            if changed:
                viewer._reload_retargeter(new_config, [])
                viewer.retargeter.reset()

            viewer._check_highlight_timeout()

            # Get live data
            try:
                fingers_data = rs_device.get_fingers_data()
                raw_kp = fingers_data[hand_key]

                if not np.allclose(raw_kp, 0):
                    last_result = viewer._process_frame(raw_kp)
                    viewer.data.qpos[:] = last_result["qpos"]
                    mujoco.mj_forward(viewer.model, viewer.data)
            except Exception:
                pass

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

    if hasattr(rs_device, "stop"):
        rs_device.stop()
    if hasattr(rs_device, "cleanup"):
        rs_device.cleanup()

    print("\nViewer closed")


def run_zed_mode(args):
    """Run tuning tool with ZED camera (live mode)."""
    try:
        from input_devices.zed_mediapipe import ZedMediaPipe
    except ImportError:
        print("Error: zed mode requires pyzed, mediapipe, and opencv-python")
        print("Install pyzed with: python /usr/local/zed/get_python_api.py")
        sys.exit(1)

    import mujoco
    import mujoco.viewer
    import signal

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    # Create viewer
    viewer = TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(config_path),
        viz_config_path=str(Path(__file__).parent / args.viz_config) if args.viz_config else None,
    )

    # Create ZED device
    zed_device = ZedMediaPipe(
        hand_side=args.hand,
        video_config=video_config,
        show_video=args.show_video,
    )

    hand_key = f"{args.hand}_fingers"
    running = True

    def signal_handler(_sig, _frame):
        nonlocal running
        running = False

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    print(f"Tuning Viewer - ZED Live Mode")
    print(f"  Config: {args.config}")
    print(f"  Hand: {args.hand}")
    print(f"  Edit config file to tune parameters (auto-reload)")
    print(f"{'=' * 50}")

    # Initialize model
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
            # Check config
            changed, new_config = viewer.config_watcher.check()
            if changed:
                viewer._reload_retargeter(new_config, [])
                viewer.retargeter.reset()

            viewer._check_highlight_timeout()

            # Get live data
            try:
                fingers_data = zed_device.get_fingers_data()
                raw_kp = fingers_data[hand_key]

                if not np.allclose(raw_kp, 0):
                    last_result = viewer._process_frame(raw_kp)
                    viewer.data.qpos[:] = last_result["qpos"]
                    mujoco.mj_forward(viewer.model, viewer.data)
            except Exception:
                pass

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

    if hasattr(zed_device, "cleanup"):
        zed_device.cleanup()

    print("\nViewer closed")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Tuning Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Skeleton Colors:
  Orange  = Raw MediaPipe input keypoints
  Cyan    = Scaled target keypoints (after segment_scaling)
  White   = Robot FK keypoints (actual robot positions)

Workflow:
  1. Start the viewer with a recording or live input
  2. Open the retarget config YAML in your editor
  3. Modify parameters (e.g., segment_scaling, lp_alpha)
  4. Save the file - viewer auto-reloads and shows changes
  5. Compare the three skeleton layers to evaluate tuning

Examples:
  mjpython tuning_tool.py --play data/avp1.pkl --hand left
  mjpython tuning_tool.py --video data/right.mp4 --hand right
  mjpython tuning_tool.py --realsense --hand right
  mjpython tuning_tool.py --zed --hand right
        """,
    )

    parser.add_argument("--config", type=str, default="config/adaptive_analytical_avp.yaml",
                        help="Path to retarget YAML config (default: config/adaptive_analytical_avp.yaml)")
    parser.add_argument("--viz-config", type=str, default=None,
                        help="Path to visualization config (default: config/tuning_viz.yaml)")
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
    parser.add_argument("--show-video", action="store_true",
                        help="Also show 2D video overlay (camera live mode)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Playback FPS (default: 30)")

    args = parser.parse_args()

    # Auto-switch config for video/realsense/zed
    if (args.video or args.realsense or args.zed) and args.config == "config/adaptive_analytical_avp.yaml":
        args.config = "config/adaptive_analytical_video.yaml"

    # Default viz config
    if args.viz_config is None:
        default_viz = Path(__file__).parent / "config" / "tuning_viz.yaml"
        if default_viz.exists():
            args.viz_config = "config/tuning_viz.yaml"

    # Determine mode
    if args.zed:
        run_zed_mode(args)
    elif args.realsense:
        run_realsense_mode(args)
    elif args.video:
        run_video_mode(args)
    elif args.play:
        run_recording_mode(args)
    else:
        # Default: play example recording
        args.play = "data/avp1.pkl"
        run_recording_mode(args)


if __name__ == "__main__":
    main()
