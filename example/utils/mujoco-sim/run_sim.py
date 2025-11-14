#!/usr/bin/env python3
import mujoco
import mujoco.viewer
from pathlib import Path
import time
import numpy as np


def main():
    side = "right"
    trajectory_path = "data/wave.npy"

    mjcf_path = (Path(__file__).parent / "model" / f"{side}.xml").resolve()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i]
            data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            data.ctrl[i] = 0.0

    for _ in range(100):
        mujoco.mj_step(model, data)

    traj_path = (Path(__file__).parent / trajectory_path).resolve()
    trajectory = np.load(traj_path)
    frame_idx = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.05]
        
        while viewer.is_running():
            data.ctrl[:] = trajectory[frame_idx % len(trajectory)]
            frame_idx += 1
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()