# WujiHand MuJoCo Simulation and Control

A minimal demo for loading and controlling the WujiHand model in the MuJoCo simulator.


https://github.com/user-attachments/assets/4b3d6d5c-420e-4e15-bbe7-68bcad9729f0

<video src="./assets/video.mp4" controls=""></video>

## Requirements

* Python 3.8+ (recommend tested environment)
* Python packages: `pip install -r requirements.txt`

## Quick start

Run the simulation with trajectory playback:
```bash
python run_sim.py
```

The script loads the default right hand model and plays the trajectory from `data/wave.npy` in a loop. To use the left hand, edit `side = "left"` in `run_sim.py`.
