"""
DexRetargeting Teleoperation Controller

Teleoperation controller using DexRetargeting for hand retargeting.
Supports different input/output device combinations.
"""

import sys
from pathlib import Path
import wujihandpy
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_devices import VisionPro, VisionProReplay
from utils.avp_utils import retarget_vision_pro


def run_teleop(
    hand_side: str = "right",
    input_device_type: str = "visionpro_replay",
    visionpro_record_path: str = "record_example.pkl",
    visionpro_ip: str = "192.168.50.127"
):
    """Run teleoperation controller."""
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"
    
    # Initialize hand with filter
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    handcontroller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
    )
    time.sleep(0.5)
    
    # Initialize input device
    device_map = {
        "visionpro_real": lambda: VisionPro(ip=visionpro_ip),
        "visionpro_replay": lambda: VisionProReplay(record_path=visionpro_record_path),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")
    input_device = device_map[input_device_type]()
    
    try:
        while True:
            # Get finger data
            fingers_data = input_device.get_fingers_data()
            fingers_mat = fingers_data[f"{hand_side}_fingers"]  # (25, 4, 4)
            
            # Retarget using AVP utility
            retarget_result = retarget_vision_pro(fingers_mat, hand_side=hand_side)
            
            # Reshape to (5, 4) and send to hand (using controller with filter)
            hand_positions = retarget_result.robot_qpos.reshape(5, 4)
            handcontroller.set_joint_target_position(hand_positions)
            
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        # Cleanup
        hand.write_joint_enabled(False)
        # input_device.cleanup()

if __name__ == "__main__":
    config = {
        "hand_side": "left", # "right" | "left"
        "input_device_type": "visionpro_replay",  # "visionpro_real" | "visionpro_replay"
        "visionpro_record_path": "record_example.pkl",
        "visionpro_ip": "192.168.50.127",
    }
    run_teleop(**config)
