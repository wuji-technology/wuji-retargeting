"""
DexRetargeting Teleoperation Controller - Simulation

Teleoperation controller using DexRetargeting for hand retargeting.
Uses MuJoCo simulation instead of real hardware.
"""

import sys
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import time
from wuji_retargeting import WujiHandRetargeter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_devices import VisionPro, VisionProReplay


def run_teleop(
    hand_side: str = "right",
    input_device_type: str = "visionpro_replay",
    visionpro_record_path: str = "record_example.pkl",
    visionpro_ip: str = "192.168.50.127",
):
    """Run teleoperation controller with MuJoCo simulation."""
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"
    
    # Load MuJoCo model
    mujoco_sim_path = Path(__file__).parent / "utils" / "mujoco-sim"
    mjcf_path = mujoco_sim_path / "model" / f"{hand_side}.xml"
    
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
        "visionpro_real": lambda: VisionPro(ip=visionpro_ip),
        "visionpro_replay": lambda: VisionProReplay(record_path=visionpro_record_path),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")
    input_device = device_map[input_device_type]()
    hand_retargeter = WujiHandRetargeter(hand_side)
    
    try:
        print("Starting teleoperation loop with MuJoCo simulation...")
        while True:
            # Get finger data
            fingers_data = input_device.get_fingers_data()
            fingers_mat = fingers_data[f"{hand_side}_fingers"]  # (25, 4, 4)
            
            # Retarget using AVP utility
            Wujihand_positions = hand_retargeter.retarget(fingers_mat)
            flat_positions = Wujihand_positions.flatten()
            # Reshape to (5, 4) and set control
            
            
            if len(flat_positions) == model.nu:
                data.ctrl[:] = flat_positions
            else:
                min_len = min(len(flat_positions), model.nu)
                data.ctrl[:min_len] = flat_positions[:min_len]
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            if viewer is not None:
                viewer.sync()
            
            time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        viewer.close()
        # input_device.cleanup()


if __name__ == "__main__":
    config = {
        "hand_side": "left", # "right" | "left"
        "input_device_type": "visionpro_replay",  # "visionpro_real" | "visionpro_replay"
        "visionpro_record_path": "record_example.pkl",
        "visionpro_ip": "192.168.50.127",
    }
    run_teleop(**config)
