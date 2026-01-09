"""Teleoperation with Real Wuji Hand Hardware.

Uses the Retargeter interface to map hand tracking input to Wuji Hand joint angles,
sent to real hardware via wujihandpy.

Usage:
    # Simple run with default (replay data/avp1.pkl)
    python teleop_real.py

    # Replay MediaPipe recording
    python teleop_real.py --play data/avp1.pkl

    # Live VisionPro input
    python teleop_real.py --input visionpro --ip <your-vision-pro-ip>

    # Record input data
    python teleop_real.py --input visionpro --record

Input device types:
- visionpro: Live VisionPro input
- mediapipe_replay: Replay recorded MediaPipe hand tracking data
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import wujihandpy

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
    visionpro_ip: str = "192.168.50.127",
    mediapipe_replay_path: str = "data/avp1.pkl",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
):
    """Run teleoperation with real hardware.

    Args:
        hand_side: 'right' or 'left'
        config_path: Path to YAML configuration file
        input_device_type: Input device type ('visionpro' or 'mediapipe_replay')
        visionpro_ip: VisionPro IP address
        mediapipe_replay_path: Path to MediaPipe recording (.pkl)
        playback_speed: Playback speed for replay mode
        playback_loop: Whether to loop replay
        enable_recording: Whether to record raw input data
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    # Initialize hardware
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    handcontroller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=5.0)
    )
    time.sleep(0.5)

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

        while True:
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

            # Send to hardware
            handcontroller.set_joint_target_position(qpos.reshape(5, 4))


    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        hand.write_joint_enabled(False)

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description='Teleoperation with Real Wuji Hand Hardware',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple run with default (replay data/avp1.pkl)
  python teleop_real.py

  # Replay MediaPipe recording
  python teleop_real.py --play data/avp1.pkl

  # Live VisionPro input
  python teleop_real.py --input visionpro --ip <your-vision-pro-ip>

  # Record input data while using VisionPro
  python teleop_real.py --input visionpro --record
        """
    )

    # Config
    parser.add_argument('--config', type=str, default='config/adaptive_analytical_avp.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--hand', type=str, default='right', choices=['left', 'right'],
                        help='Hand side (default: right)')

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

    # Default to mediapipe_replay with data/avp1.pkl if no input specified
    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    # Run teleoperation
    log = run_teleop(
        hand_side=args.hand,
        config_path=args.config,
        input_device_type=input_device_type,
        visionpro_ip=args.ip,
        mediapipe_replay_path=mediapipe_replay_path,
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
