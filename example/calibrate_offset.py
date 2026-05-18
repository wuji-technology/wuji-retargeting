#!/usr/bin/env python3
"""Calibrate wrist_offset_cm and thumb_offset_cm for wuji_glove.

Compares MediaPipe MCP positions with robot model MCP positions at default qpos
to compute the bias needed to align them.

Usage (live from glove, default):
    python3 example/calibrate_offset.py --hand right \\
        --config config/adaptive_analytical_wuji_glove_right.yaml

    # Save the sampled frames for later replay:
    python3 example/calibrate_offset.py --hand right \\
        --dump-pkl data/my_calib_take.pkl

Usage (replay from pkl):
    python3 example/calibrate_offset.py \\
        --play data/wuji_glove_record.pkl --hand right \\
        --config config/adaptive_analytical_wuji_glove_right.yaml
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting import Retargeter
from wuji_retargeting.mediapipe import apply_mediapipe_transformations


# MediaPipe indices for MCP joints
FOUR_FINGER_MCP = [5, 9, 13, 17]    # index, middle, ring, pinky
THUMB_CMC = 1                         # thumb CMC (closest to wrist)

# Robot link names for MCP joints (same order as above)
FOUR_FINGER_ROBOT_MCP = [
    "finger2_link1",  # index MCP
    "finger3_link1",  # middle MCP
    "finger4_link1",  # ring MCP
    "finger5_link1",  # pinky MCP
]
THUMB_ROBOT_CMC = "finger1_link1"     # thumb CMC


def get_robot_mcp_positions(retargeter):
    """Get robot MCP positions at default qpos (all zeros) in MANO frame.

    Returns positions relative to palm_link, rotated into MANO frame.
    """
    robot = retargeter.optimizer.robot

    # Use zero qpos (default/rest pose)
    qpos = np.zeros(retargeter.num_joints, dtype=np.float64)
    robot.compute_forward_kinematics(qpos)

    # Get palm (wrist) position
    palm_idx = robot.get_link_index("palm_link")
    palm_pose = robot.get_link_pose(palm_idx)
    palm_pos = palm_pose[:3, 3]

    # Get four-finger MCP positions relative to palm, in MANO frame
    four_finger_pos = []
    for link_name in FOUR_FINGER_ROBOT_MCP:
        idx = robot.get_link_index(link_name)
        link_pos = robot.get_link_pose(idx)[:3, 3]
        # Pinocchio poses are in URDF-root frame, which is MANO-aligned for
        # supported Wuji hands, so link_pos - palm_pos is already a MANO-frame
        # vector. No further rotation needed.
        rel = link_pos - palm_pos
        four_finger_pos.append(rel)

    # Get thumb CMC position (MANO-frame difference, see note above)
    thumb_idx = robot.get_link_index(THUMB_ROBOT_CMC)
    thumb_pos = robot.get_link_pose(thumb_idx)[:3, 3]
    thumb_rel = thumb_pos - palm_pos

    return np.array(four_finger_pos), thumb_rel


def compute_mediapipe_mcp_positions(raw_keypoints, hand_side, rotation_xyz):
    """Compute MediaPipe MCP positions in MANO frame (relative to wrist=0)."""
    from scipy.spatial.transform import Rotation

    kp = apply_mediapipe_transformations(raw_keypoints, hand_side)

    # Apply mediapipe_rotation if configured
    x_deg = rotation_xyz.get('x', 0.0)
    y_deg = rotation_xyz.get('y', 0.0)
    z_deg = rotation_xyz.get('z', 0.0)
    if x_deg != 0 or y_deg != 0 or z_deg != 0:
        rot = Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True)
        kp = kp @ rot.as_matrix().T

    # kp[0] = 0 (wrist at origin)
    four_finger_mcp = kp[FOUR_FINGER_MCP]  # (4, 3)
    thumb_cmc = kp[THUMB_CMC]              # (3,)

    return four_finger_mcp, thumb_cmc


def collect_frames_from_pkl(
    pkl_path: Path, limit: int | None, trust_pkl: bool = False
):
    if not trust_pkl:
        raise ValueError(
            "Refusing to load pickle without explicit trust. "
            "Use --trust-pkl only for files you fully trust."
        )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} frames from {pkl_path.name}")
    if limit:
        data = data[:limit]
        print(f"Using first {len(data)} frames")
    return data


def collect_frames_live(hand_side: str, device_name: str, glove_sn: str | None,
                        num_frames: int, rate_hz: float):
    """Sample ``num_frames`` live frames from ``wuji_glove``."""
    from input_devices.wuji_glove_device import WujiGloveDevice

    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be > 0, got {rate_hz}")

    print(f"[live] Connecting glove device_name={device_name} sn={glove_sn or 'auto'}")
    device = WujiGloveDevice(
        hand_side=hand_side, device_name=device_name, sn=glove_sn,
    )

    hand_key = f"{hand_side}_fingers"
    period = 1.0 / rate_hz
    frames: list[dict] = []
    skipped = 0
    last_kp_id = None
    try:
        print(
            f"\nPlace your {hand_side} hand in the calibration pose "
            f"(palm flat, fingers relaxed, straight, and together), "
            f"then press Enter to sample {num_frames} frames..."
        )
        input()

        t0 = time.perf_counter()
        while len(frames) < num_frames:
            loop_t = time.perf_counter()
            data = device.get_fingers_data()
            kp = data.get(hand_key)
            kp_id = id(kp)
            if kp is None or np.allclose(kp, 0):
                skipped += 1
            elif kp_id == last_kp_id:
                # same buffer as last frame (SDK no new data) ; skip to avoid dup
                pass
            else:
                frames.append({hand_key: np.array(kp, dtype=np.float32)})
                last_kp_id = kp_id
                if len(frames) % 10 == 0 or len(frames) == num_frames:
                    print(f"  sampled {len(frames)}/{num_frames}")
            elapsed = time.perf_counter() - loop_t
            if elapsed < period:
                time.sleep(period - elapsed)
            if time.perf_counter() - t0 > 30.0 and len(frames) == 0:
                print("[live] No valid frame received within 30s. Check the glove connection.")
                break
    finally:
        device.cleanup()

    print(f"[live] Sampling complete: {len(frames)} frames (skipped {skipped} empty/duplicate)")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Calibrate wrist/thumb offset for wuji_glove")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--play", type=str, default=None,
                     help="Replay mode: path to a recorded .pkl file")
    src.add_argument("--live", action="store_true",
                     help="Live mode (default): sample N frames from the glove")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--config", type=str,
                        default="config/adaptive_analytical_wuji_glove_right.yaml",
                        help="Retarget config YAML")
    parser.add_argument("--frames", type=int, default=None,
                        help="Replay mode: use only the first N frames")
    parser.add_argument("--trust-pkl", action="store_true",
                        help="Explicitly confirm that the --play .pkl file comes from a trusted source")
    parser.add_argument("--num-frames", type=int, default=60,
                        help="Number of live samples to capture (default: 60)")
    parser.add_argument("--rate-hz", type=float, default=60.0,
                        help="Live sampling rate (default: 60Hz)")
    parser.add_argument("--device-name", type=str, default="glove",
                        help="SDK device_name")
    parser.add_argument("--glove-sn", type=str, default=None,
                        help="Glove serial number; optional when only one device is connected")
    parser.add_argument("--dump-pkl", type=str, default=None,
                        help="Save raw live frames to this .pkl file for later replay")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config

    # Collect frames
    if args.play:
        pkl_path = script_dir / args.play
        data = collect_frames_from_pkl(
            pkl_path, args.frames, trust_pkl=args.trust_pkl
        )
    else:
        data = collect_frames_live(
            hand_side=args.hand,
            device_name=args.device_name,
            glove_sn=args.glove_sn,
            num_frames=args.num_frames,
            rate_hz=args.rate_hz,
        )
        if args.dump_pkl:
            dump_path = script_dir / args.dump_pkl
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[live] Raw frames saved to {dump_path}")

    if not data:
        print("ERROR: no frames collected!")
        return

    # Load retargeter (for robot model + config)
    retargeter = Retargeter.from_yaml(str(config_path), hand_side=args.hand)
    rotation_xyz = retargeter.rotation_xyz

    # Get robot MCP positions at default pose
    robot_four_mcp, robot_thumb_cmc = get_robot_mcp_positions(retargeter)
    print("\nRobot MCP positions at default qpos (meters, MANO frame):")
    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        print(f"  {name:7s} MCP: [{robot_four_mcp[i][0]:+.5f}, {robot_four_mcp[i][1]:+.5f}, {robot_four_mcp[i][2]:+.5f}]")
    print(f"  thumb   CMC: [{robot_thumb_cmc[0]:+.5f}, {robot_thumb_cmc[1]:+.5f}, {robot_thumb_cmc[2]:+.5f}]")

    # Compute MediaPipe MCP positions across all frames
    hand_key = f"{args.hand}_fingers"
    four_finger_diffs = []
    thumb_diffs = []
    skipped = 0

    for frame in data:
        raw_kp = frame.get(hand_key)
        if raw_kp is None or np.allclose(raw_kp, 0):
            skipped += 1
            continue

        mp_four_mcp, mp_thumb_cmc = compute_mediapipe_mcp_positions(
            raw_kp, args.hand, rotation_xyz
        )

        # Diff = robot - mediapipe (this is the bias to add to mediapipe to reach robot)
        four_finger_diffs.append(robot_four_mcp - mp_four_mcp)  # (4, 3)
        thumb_diffs.append(robot_thumb_cmc - mp_thumb_cmc)       # (3,)

    if not four_finger_diffs:
        print("ERROR: no valid frames found!")
        return

    print(f"\nProcessed {len(four_finger_diffs)} frames (skipped {skipped})")

    # Average across frames, then average across 4 fingers
    four_finger_diffs = np.array(four_finger_diffs)   # (N, 4, 3)
    thumb_diffs = np.array(thumb_diffs)                # (N, 3)

    # Per-finger average across frames
    per_finger_avg = four_finger_diffs.mean(axis=0)    # (4, 3)
    print(f"\nPer-finger MCP offset (robot - mediapipe, meters):")
    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        d = per_finger_avg[i]
        print(f"  {name:7s}: [{d[0]:+.5f}, {d[1]:+.5f}, {d[2]:+.5f}]")

    # Global four-finger offset = average of all 4 fingers
    wrist_offset_m = per_finger_avg.mean(axis=0)       # (3,)
    wrist_offset_cm = wrist_offset_m * 100.0

    # Thumb offset
    thumb_offset_m = thumb_diffs.mean(axis=0)           # (3,)
    thumb_offset_cm = thumb_offset_m * 100.0

    print(f"\n{'='*60}")
    print(f"Recommended YAML values:")
    print(f"{'='*60}")
    print(f"  wrist_offset_cm: [{wrist_offset_cm[0]:.2f}, {wrist_offset_cm[1]:.2f}, {wrist_offset_cm[2]:.2f}]")
    print(f"  thumb_offset_cm: [{thumb_offset_cm[0]:.2f}, {thumb_offset_cm[1]:.2f}, {thumb_offset_cm[2]:.2f}]")
    print(f"{'='*60}")

    # Also show per-finger residual after applying the global offset
    print(f"\nPer-finger residual after applying wrist_offset (cm):")
    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        residual = (per_finger_avg[i] - wrist_offset_m) * 100
        print(f"  {name:7s}: [{residual[0]:+.3f}, {residual[1]:+.3f}, {residual[2]:+.3f}]")


if __name__ == "__main__":
    main()
