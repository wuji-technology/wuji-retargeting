"""Teleoperation with MuJoCo Simulation.

Uses the Retargeter interface to map hand tracking input to Wuji Hand joint angles,
visualized in MuJoCo simulation.

Usage:
    # Replay MediaPipe recording (default)
    mjpython teleop_sim.py --play data/avp1.pkl --hand left

    # Live VisionPro input
    mjpython teleop_sim.py --input visionpro --ip <your-vision-pro-ip>

    # Record input data while using VisionPro
    mjpython teleop_sim.py --input visionpro --record

Input device types:
- visionpro: Live VisionPro input
- mediapipe_replay: Replay recorded MediaPipe hand tracking data
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting import Retargeter
from input_devices.visionpro import VisionPro
from input_devices.mediapipe_replay import MediaPipeReplay


def run_teleop(
    hand_side: str = "right",
    config_path: str = "config/adaptive_analytical_avp.yaml",
    input_device_type: str = "mediapipe_replay",
    mediapipe_replay_path: str = "",
    visionpro_ip: str = "192.168.50.127",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
):
    """Run teleoperation with MuJoCo simulation.

    Args:
        hand_side: 'right' or 'left'
        config_path: Path to YAML configuration file
        input_device_type: Input device type ('visionpro' or 'mediapipe_replay')
        mediapipe_replay_path: Path to MediaPipe recording (.pkl)
        visionpro_ip: VisionPro IP address
        playback_speed: Playback speed for replay mode
        playback_loop: Whether to loop replay
        enable_recording: Whether to record raw input data
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    # Load MuJoCo model
    mujoco_sim_path = Path(__file__).parent / "utils" / "mujoco-sim"
    mjcf_path = mujoco_sim_path / "wuji_hand_description" / "mjcf" / f"{hand_side}.xml"
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # Initialize control signals
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i]
            data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            data.ctrl[i] = 0.0

    # Stabilize model
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0, 0, 0.05]

    # Initialize input device
    device_map = {
        "visionpro": lambda: VisionPro(ip=visionpro_ip),
        "mediapipe_replay": lambda: MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        ),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        raise ValueError("mediapipe_replay_path is required for mediapipe_replay mode")

    input_device = device_map[input_device_type]()

    # Initialize retargeter
    config_file = Path(__file__).parent / config_path
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # Disable recording when using replay mode
    if input_device_type == "mediapipe_replay" and enable_recording:
        print("Note: Recording disabled in replay mode")
        enable_recording = False

    # Prepare recording
    input_data_log = [] if enable_recording else None
    start_time = time.time()

    try:
        print(f"Starting teleoperation...")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        print("=" * 50)

        frame_count = 0
        fps_start_time = time.time()

        while viewer.is_running():
            # Get finger data
            fingers_data = input_device.get_fingers_data()
            fingers_pose = fingers_data[f"{hand_side}_fingers"]  # (21, 3)

            # Skip if data is all zeros
            if np.allclose(fingers_pose, 0):
                time.sleep(0.01)
                continue

            # Record raw input data if enabled
            if enable_recording:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": fingers_data["left_fingers"].copy(),
                    "right_fingers": fingers_data["right_fingers"].copy(),
                })

            # Retarget to joint angles
            qpos = retargeter.retarget(fingers_pose)  # (20,)

            # FPS counter
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}")

            # Set control signals
            if len(qpos) == model.nu:
                data.ctrl[:] = qpos
            else:
                min_len = min(len(qpos), model.nu)
                data.ctrl[:min_len] = qpos[:min_len]

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        viewer.close()

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description='Teleoperation with MuJoCo Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay MediaPipe recording
  mjpython teleop_sim.py --play data/avp1.pkl --hand left

  # Live VisionPro input
  mjpython teleop_sim.py --input visionpro --ip <your-vision-pro-ip>

  # Record input data while using VisionPro
  mjpython teleop_sim.py --input visionpro --record
        """
    )

    # Config
    parser.add_argument('--config', type=str, default='config/adaptive_analytical_avp.yaml',
                        help='Path to YAML configuration file (default: config/adaptive_analytical_avp.yaml)')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'],
                        help='Hand side (default: left)')

    # Input device options
    parser.add_argument('--input', type=str, default=None,
                        choices=['visionpro', 'mediapipe_replay'],
                        help='Input device type')

    # Shortcut options
    parser.add_argument('--play', type=str, default=None, metavar='FILE',
                        help='Play MediaPipe recording file (shortcut for --input mediapipe_replay)')

    # VisionPro options
    parser.add_argument('--ip', type=str, default='192.168.50.127',
                        help='VisionPro IP address (default: 192.168.50.127)')

    # Playback options
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed for replay mode (default: 1.0)')
    parser.add_argument('--no-loop', action='store_true',
                        help='Disable looping for replay mode')

    # Recording
    parser.add_argument('--record', action='store_true',
                        help='Record input data to file')
    parser.add_argument('--output', type=str, default=None, metavar='FILE',
                        help='Output file for recording (default: auto-generated)')

    args = parser.parse_args()

    # Determine input device type and paths
    input_device_type = args.input
    mediapipe_replay_path = ""

    if args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    # Default to mediapipe_replay with example data if no input specified
    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    # Validate paths
    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        parser.error("--play FILE is required for mediapipe_replay mode")

    # Run teleoperation
    log = run_teleop(
        hand_side=args.hand,
        config_path=args.config,
        input_device_type=input_device_type,
        mediapipe_replay_path=mediapipe_replay_path,
        visionpro_ip=args.ip,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=args.record,
    )

    # Save recording if enabled
    if log is not None and len(log) > 0:
        if args.output:
            log_path = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"

        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()
