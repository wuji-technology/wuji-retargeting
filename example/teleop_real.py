"""Teleoperation with Real Wuji Hand Hardware.

Uses the Retargeter interface to map hand tracking input to Wuji Hand joint angles,
sent to real hardware via wujihandpy.

Usage:
    # Simple run with default (replay data/avp1.pkl)
    python teleop_real.py

    # Replay input data file
    python teleop_real.py --play data/avp1.pkl

    # Live VisionPro input
    python teleop_real.py --input visionpro_real --ip <your-vision-pro-ip>

    # Live Manus glove input
    python teleop_real.py --input manus_glove --glove-id 0

    # Record input data
    python teleop_real.py --input visionpro_real --record

Input device types:
- visionpro_real: Live VisionPro input
- manus_glove: Live Manus glove input
- input_data_replay: Replay recorded raw input data (MediaPipe format)
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
from input_devices import VisionPro, InputDataReplay


def run_teleop(
    hand_side: str = "right",
    config_path: str = "config/adaptive_analytical_avp.yaml",
    input_device_type: str = "input_data_replay",
    visionpro_ip: str = "192.168.50.127",
    manus_glove_id: int = 0,
    input_data_replay_path: str = "data/avp1.pkl",
    input_data_replay_speed: float = 1.0,
    input_data_replay_loop: bool = True,
    enable_recording: bool = False,
):
    """Run teleoperation with real hardware.

    Args:
        hand_side: 'right' or 'left'
        config_path: Path to YAML configuration file
        input_device_type: Input device type
        visionpro_ip: VisionPro IP address
        manus_glove_id: Manus glove ID
        input_data_replay_path: Path to input data recording (.pkl)
        input_data_replay_speed: Playback speed for input data replay
        input_data_replay_loop: Whether to loop input data replay
        enable_recording: Whether to record raw input data
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    # Initialize hardware
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    handcontroller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=3.0)
    )
    time.sleep(0.5)

    # Initialize input device
    def create_manus_device():
        from input_devices.manus_glove import ManusGloveDevice
        return ManusGloveDevice(glove_id=manus_glove_id)

    device_map = {
        "visionpro_real": lambda: VisionPro(ip=visionpro_ip),
        "manus_glove": create_manus_device,
        "input_data_replay": lambda: InputDataReplay(
            record_path=input_data_replay_path,
            playback_speed=input_data_replay_speed,
            loop=input_data_replay_loop,
        ),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    if input_device_type == "input_data_replay" and not input_data_replay_path:
        raise ValueError("input_data_replay_path is required for input_data_replay mode")

    input_device = device_map[input_device_type]()

    # Initialize retargeter
    config_file = Path(__file__).parent / config_path
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # Disable recording when using replay mode
    if input_device_type == "input_data_replay" and enable_recording:
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

            time.sleep(0.005)

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

  # Replay input data file
  python teleop_real.py --play data/avp1.pkl

  # Live VisionPro input
  python teleop_real.py --input visionpro_real --ip <your-vision-pro-ip>

  # Live Manus glove input
  python teleop_real.py --input manus_glove --glove-id 0

  # Record input data while using VisionPro
  python teleop_real.py --input visionpro_real --record
        """
    )

    # Config
    parser.add_argument('--config', type=str, default='config/adaptive_analytical_avp.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--hand', type=str, default='right', choices=['left', 'right'],
                        help='Hand side (default: right)')

    # Input device options
    parser.add_argument('--input', type=str, default=None,
                        choices=['visionpro_real', 'manus_glove', 'input_data_replay'],
                        help='Input device type')

    # Shortcut options
    parser.add_argument('--play', type=str, default=None, metavar='FILE',
                        help='Play input data file (shortcut for --input input_data_replay)')

    # VisionPro options
    parser.add_argument('--ip', type=str, default='192.168.50.127',
                        help='VisionPro IP address (default: 192.168.50.127)')

    # Manus glove options
    parser.add_argument('--glove-id', type=int, default=0,
                        help='Manus glove ID (default: 0)')

    # Playback options
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed for replay modes (default: 1.0)')
    parser.add_argument('--no-loop', action='store_true',
                        help='Disable looping for replay modes')

    # Recording
    parser.add_argument('--record', action='store_true',
                        help='Record input data to file')
    parser.add_argument('--output', type=str, default=None, metavar='FILE',
                        help='Output file for recording (default: auto-generated)')

    args = parser.parse_args()

    # Determine input device type and paths
    input_device_type = args.input
    input_data_replay_path = ""

    if args.play:
        input_device_type = "input_data_replay"
        input_data_replay_path = args.play

    # Default to input_data_replay with data/avp1.pkl if no input specified
    if input_device_type is None:
        input_device_type = "input_data_replay"
        input_data_replay_path = "data/avp1.pkl"

    # Run teleoperation
    log = run_teleop(
        hand_side=args.hand,
        config_path=args.config,
        input_device_type=input_device_type,
        visionpro_ip=args.ip,
        manus_glove_id=args.glove_id,
        input_data_replay_path=input_data_replay_path,
        input_data_replay_speed=args.speed,
        input_data_replay_loop=not args.no_loop,
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
