"""Teleoperation with Real Wuji Hand Hardware.

Uses the Retargeter interface to map hand tracking input to Wuji Hand joint angles,
sent to real hardware via wujihandpy.

Usage:
    # Simple run with default (replay data/avp1.pkl)
    python teleop_real.py

    # Replay MediaPipe recording
    python teleop_real.py --play data/avp1.pkl

    # MP4 video input with MediaPipe hand detection
    python teleop_real.py --video data/right.mp4 --hand right
    python teleop_real.py --video data/right.mp4 --hand right --show-video

    # RealSense camera input with MediaPipe hand detection
    python teleop_real.py --realsense --hand right

    # ZED camera input with MediaPipe hand detection
    python teleop_real.py --zed --hand right

    # Live VisionPro input
    python teleop_real.py --input visionpro --ip <your-vision-pro-ip>

    # Record input data
    python teleop_real.py --input visionpro --record

    # Live Wuji Glove input via wuji_sdk
    python teleop_real.py --input wuji_glove --hand right --glove-sn <YOUR_SN>

Input device types:
- visionpro: Live VisionPro input
- mediapipe_replay: Replay recorded MediaPipe hand tracking data
- video: MP4 video input with MediaPipe hand detection
- realsense: RealSense camera input with MediaPipe hand detection
- zed: ZED camera input with MediaPipe hand detection
- wuji_glove: Live Wuji Glove input via wuji_sdk
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import wujihandpy
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting import Retargeter
from utils.config_paths import resolve_mjcf_path, mjcf_joint_order, qpos_reorder_perm
from input_devices.visionpro import VisionPro
from input_devices.mediapipe_replay import MediaPipeReplay
try:
    from input_devices.video_mediapipe import VideoMediaPipe
except ImportError:
    VideoMediaPipe = None
try:
    from input_devices.realsense_mediapipe import RealsenseMediaPipe
except ImportError:
    RealsenseMediaPipe = None
try:
    from input_devices.zed_mediapipe import ZedMediaPipe
except ImportError:
    ZedMediaPipe = None
try:
    from input_devices.wuji_glove_device import WujiGloveDevice
    WUJI_SDK_AVAILABLE = True
except ImportError:
    WujiGloveDevice = None
    WUJI_SDK_AVAILABLE = False


# =============================================================================
# Hardware backends.
#
# WH110 connects over USB-CDC via wujihandpy.Hand; WH120 is a networked
# "Wuji Hand 2" reached over Ethernet via wuji_sdk.SdkManager. Both are wrapped
# behind a uniform { send(qpos), close() } interface so the teleop loop dispatches
# on --hand-model without caring about the transport. close() is called from a
# finally block so Ctrl+C also runs cleanup and the WH120 session is not leaked.
# =============================================================================
from typing import Optional


# NOTE: per-joint calibration (sign/offset) is intentionally NOT exposed in the
# public release. Feeding an unvalidated sign flip / offset straight to the MIT
# controller can drive a real joint in the wrong direction or past its safe range,
# and it cannot be previewed in simulation — too dangerous for an end-user knob.
# WH120 runs with the validated default joint mapping (firmware == retargeted qpos).


def _set_with_retry(desc, fn, attempts=3, backoff=0.6):
    """Call fn() (a device-config SET), retrying on transient SDK timeouts.

    Right after connect — especially when a handedness probe just churned the
    shared 'wuji_hand_2' bridge by connecting/disconnecting another hand (only
    happens with 2+ hands online) — the first config SET can hit "device not
    responding" while the old bridge is still tearing down. These SETs are
    idempotent, so a few short retries recover it. Bounded (default 3 attempts)
    so a genuinely dead hand still fails fast, and the success path takes the
    first attempt with no added delay.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            print(f"WH120 init: {desc} not responding "
                  f"(attempt {i + 1}/{attempts}); bridge settling, retrying in {backoff}s...")
            time.sleep(backoff)


class Wh110Backend:
    """Hardware backend for WH110 via wujihandpy (USB-CDC)."""

    def __init__(self, hand_serial: str = ""):
        self._hand = (
            wujihandpy.Hand(serial_number=hand_serial) if hand_serial else wujihandpy.Hand()
        )
        self._hand.write_joint_enabled(True)
        self._controller = self._hand.realtime_controller(
            enable_upstream=False,
            filter=wujihandpy.filter.LowPass(cutoff_freq=5.0),
        )
        time.sleep(0.5)

    def send(self, qpos: np.ndarray) -> None:
        self._controller.set_joint_target_position(qpos.reshape(5, 4))

    def close(self) -> None:
        if getattr(self, "_hand", None) is not None:
            self._hand.write_joint_enabled(False)


class Wh120Backend:
    """Hardware backend for WH120 via wuji_sdk (networked Wuji Hand 2)."""

    _ENABLE_TIMEOUT_SEC = 5.0
    _INVERTER_READY_STATE = 4

    def __init__(
        self,
        ip: str,
        kp: float,
        kd: float,
        current_limit: float,
        handedness: Optional[str] = None,
    ):
        try:
            import wuji_sdk
            from wuji_sdk import SdkManager
        except ImportError as e:
            raise RuntimeError(
                "wuji_sdk not importable. Install the WH120 SDK wheel."
            ) from e

        self._sdk = wuji_sdk
        self._manager = SdkManager.instance()

        # Connection: explicit --wh120-ip wins. Otherwise scan for Wuji Hand 2
        # devices (SN starts "WH"; gloves are "WG"). One hand -> connect it. Several
        # hands online (bimanual) -> pick the one whose reported handedness matches
        # --hand. We can't use connect(handedness=) directly (it also matches the
        # same-side glove), and DiscoveredDevice carries no handedness, so we probe:
        # connect each hand, read hand.handedness(), keep the match, disconnect the
        # rest. Probing only opens a session (no enable / motion).
        t0 = time.monotonic()
        if ip:
            print(f"WH120 init: connect by address ip={ip}")
            self._hand = self._manager.connect(address=ip, device_name="wuji_hand_2")
        else:
            print("WH120 init: no --wh120-ip; scanning for a wuji_hand_2 on the network...")
            hands = [d for d in self._manager.scan() if str(d.sn).upper().startswith("WH")]
            if not hands:
                raise RuntimeError("no Wuji Hand 2 found on the network; check power/network.")
            if len(hands) == 1:
                d = hands[0]
                print(f"WH120 init: discovered {d.sn} at {d.address}")
                self._hand = self._manager.connect(address=d.address, device_name="wuji_hand_2")
            else:
                want = (handedness or "").lower()
                if want not in ("left", "right"):
                    listing = ", ".join(f"{d.sn}@{d.address}" for d in hands)
                    raise RuntimeError(
                        f"{len(hands)} hands online ({listing}); pass --hand left/right "
                        f"or --wh120-ip <address:port>."
                    )
                print(f"WH120 init: {len(hands)} hands online; selecting the {want} one by handedness...")
                self._hand = None
                for d in hands:
                    h = self._manager.connect(address=d.address, device_name="wuji_hand_2")
                    try:
                        side = str(h.handedness().get()).lower()
                    except Exception:
                        side = "?"
                    if side == want:
                        print(f"WH120 init: matched {want} hand {d.sn} at {d.address}")
                        self._hand = h
                        break
                    print(f"WH120 init:   {d.sn} reports '{side}', skipping")
                    h.disconnect()
                    time.sleep(0.2)  # let the device_name bridge fully release
                if self._hand is None:
                    listing = ", ".join(str(d.sn) for d in hands)
                    raise RuntimeError(
                        f"no {want} Wuji Hand 2 among [{listing}]; pass --wh120-ip <address:port>."
                    )
        print(
            f"WH120 init: connect done in {time.monotonic() - t0:.3f}s "
            f"sn={self._hand.serial_number}"
        )

        # From here on the session is open. If any setup step fails, disconnect
        # before propagating — otherwise the session is left on the hand and the
        # next run fails with "Session already exists (0x0013)".
        try:
            n_online = self._hand.online_joints_count().get()
            if n_online == 0:
                raise RuntimeError("WH120: 0 joints online — check device power/network")
            print(f"WH120 connected: {n_online}/20 joints online")

            # Settle before mode change.
            time.sleep(0.5)

            # These config SETs are retried (bounded) because the first one after a
            # multi-hand handedness probe can transiently time out while the churned
            # shared bridge settles. enable() below is intentionally NOT retried.
            _set_with_retry("control_mode=mit", lambda: self._hand.control_mode().set("mit"))
            # Current limit only (per-joint, amps), per the SDK's sys_current_limit
            # resource. The separate effort_limit knob is intentionally left at the
            # firmware default — the current limit is the governing cap.
            _set_with_retry(
                "sys_current_limit",
                lambda: self._hand.sys_current_limit().set(current_limit),
            )
            _set_with_retry(
                "mit_params",
                lambda: self._hand.mit_params().set(
                    kp=[[kp] * 4 for _ in range(5)],
                    kd=[[kd] * 4 for _ in range(5)],
                ),
            )
            self._hand.enable()
            print(f"WH120 init: enable() sent")

            # Wait until all live joints reach inverter_state=4 (ready).
            deadline = time.monotonic() + self._ENABLE_TIMEOUT_SEC
            enabled = False
            last_diags = []
            while time.monotonic() < deadline:
                time.sleep(0.2)
                last_diags = self._hand.diagnostics().get()
                # Filter on vbus > 0.5 (live inverter) instead of `not None`
                # (encoder-reachable). A joint with dead inverter still reports
                # diagnostics with vbus=0 / inverter_state=0 and would otherwise
                # block the ready check forever.
                live = [d for d in last_diags if d is not None and d.vbus > 0.5]
                if live and all(
                    d.inverter_state == self._INVERTER_READY_STATE for d in live
                ):
                    enabled = True
                    break
            if not enabled:
                print("WH120: enable timeout. Per-joint state:")
                for i, d in enumerate(last_diags):
                    if d is not None:
                        fi, ji = divmod(i, 4)
                        print(
                            f"  finger{fi + 1}/j{ji} (idx {i}): "
                            f"inverter_state={d.inverter_state} vbus={d.vbus:.2f}"
                        )
                self._hand.disable()
                raise RuntimeError(f"WH120: enable timeout after {self._ENABLE_TIMEOUT_SEC}s")
            print(f"WH120 enabled (kp={kp}, kd={kd}, current_limit={current_limit}A)")
            # publisher creation is post-enable; keep it INSIDE this guard so a
            # failure here also tears the session down instead of leaking it.
            self._publisher = self._hand.joint_command().publisher()
            self._zeros = [0.0] * 20
        except BaseException:
            # Best-effort teardown so a failed init doesn't leak the session.
            try:
                self._manager.disconnect_all()
            except Exception:
                pass
            raise

    def send(self, qpos: np.ndarray) -> None:
        # Send the retargeted qpos straight through — no per-joint calibration is
        # applied in the public path (see the module note above).
        positions = qpos.astype(np.float64).tolist()
        # MIT: positions + zero velocities + zero effort feedforward
        self._publisher.send(positions, self._zeros, self._zeros)

    def close(self) -> None:
        if getattr(self, "_publisher", None) is not None:
            self._publisher.close()
        if getattr(self, "_hand", None) is not None:
            self._hand.disable()
        if getattr(self, "_manager", None) is not None:
            self._manager.disconnect_all()


def _infer_hand_model(config_path) -> str:
    """Infer 'wh120' vs 'wh110' from a retarget config.

    A WH120 config overrides the optimizer's hand assets (``urdf_path`` /
    ``mjcf_path``) to a WH120 model, and the filename conventionally carries
    ``wh120``. Anything else is treated as the default WH110. Used only when
    ``--hand-model`` is omitted, so passing a WH120 config selects the WH120
    backend without a separate flag.
    """
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return "wh110"
    opt = cfg.get("optimizer") or {}
    blob = " ".join(
        str(x) for x in (opt.get("urdf_path"), opt.get("mjcf_path"), config_path)
    )
    return "wh120" if "wh120" in blob.lower() else "wh110"


def run_tuning_mode(
    hand_side: str,
    config_path: str,
    input_device_type: str,
    mediapipe_replay_path: str = "",
    video_path: str = "",
    show_video: bool = False,
    viz_config_path: str = None,
    fps: float = 30.0,
    device_name: str = "glove",
    glove_sn: str = "",
):
    """Backward-compatible wrapper for the standalone tuning tool."""
    import tuning_tool

    class TuningArgs:
        pass

    args = TuningArgs()
    args.hand = hand_side
    args.config = config_path
    args.viz_config = viz_config_path
    args.play = mediapipe_replay_path or None
    args.video = video_path or None
    args.realsense = input_device_type == "realsense"
    args.zed = input_device_type == "zed"
    args.wuji_glove = input_device_type == "wuji_glove"
    args.device_name = device_name
    args.glove_sn = glove_sn
    args.show_video = show_video
    args.fps = fps

    print(
        "Note: --tuning on teleop scripts is kept for compatibility. "
        "Prefer running tuning_tool.py directly."
    )

    if args.wuji_glove:
        tuning_tool.run_wuji_glove_mode(args)
    elif args.zed:
        tuning_tool.run_zed_mode(args)
    elif args.realsense:
        tuning_tool.run_realsense_mode(args)
    elif args.video:
        tuning_tool.run_video_mode(args)
    elif args.play:
        tuning_tool.run_recording_mode(args)
    else:
        print("Tuning mode requires --play FILE or a live/video input source")


def run_teleop(
    hand_side: str = "right",
    config_path: str = "config/adaptive_analytical_avp.yaml",
    input_device_type: str = "mediapipe_replay",
    visionpro_ip: str = "192.168.50.127",
    mediapipe_replay_path: str = "data/avp1.pkl",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
    video_path: str = "",
    show_video: bool = False,
    device_name: str = "glove",
    glove_sn: str = "",
    # Hardware backend selection + WH120 connection params.
    hand_model: str = "wh110",
    hand_serial: str = "",
    wh120_ip: str = "",
    kp: float = 3.0,
    kd: float = 0.1,
    current_limit: float = 1.5,
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
        video_path: Path to MP4 video file
        show_video: Whether to display video with MediaPipe landmarks overlay
        device_name: wuji_sdk device name for Wuji Glove (default "glove")
        glove_sn: Wuji Glove serial number (required when multiple Wuji devices online)
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"
    assert hand_model in {"wh110", "wh120"}, f"hand_model must be 'wh110' or 'wh120', got {hand_model}"

    # backend/input_device are set up front so the finally can always check them and
    # so a failure during setup never leaves a reference unbound. The hand is
    # energized LAST (just before the loop, inside the try) — see below.
    backend = None
    input_device = None

    # Load config to get video_input settings if needed
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get('video_input', {})

    def create_wuji_glove_device():
        if not WUJI_SDK_AVAILABLE:
            raise ImportError(
                "wuji_sdk is not installed. "
                "Please install wuji_sdk to use --input wuji_glove."
            )
        return WujiGloveDevice(
            hand_side=hand_side,
            device_name=device_name,
            sn=glove_sn or None,
        )

    # Initialize input device
    device_map = {
        "visionpro": lambda: VisionPro(ip=visionpro_ip),
        "mediapipe_replay": lambda: MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        ),
        "video": lambda: VideoMediaPipe(
            video_path=video_path,
            hand_side=hand_side,
            playback_speed=playback_speed,
            loop=playback_loop,
            video_config=video_config,
            show_video=show_video,
        ),
        "realsense": lambda: RealsenseMediaPipe(
            hand_side=hand_side,
            video_config=video_config,
            show_video=show_video,
        ),
        "zed": lambda: ZedMediaPipe(
            hand_side=hand_side,
            video_config=video_config,
            show_video=show_video,
        ),
        "wuji_glove": create_wuji_glove_device,
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        raise ValueError("mediapipe_replay_path is required for mediapipe_replay mode")
    if input_device_type == "video" and not video_path:
        raise ValueError("video_path is required for video mode")
    if input_device_type == "video" and VideoMediaPipe is None:
        raise ImportError("video mode requires mediapipe and opencv-python")
    if input_device_type == "realsense" and RealsenseMediaPipe is None:
        raise ImportError("realsense mode requires mediapipe, opencv-python, and pyrealsense2")
    if input_device_type == "zed" and ZedMediaPipe is None:
        raise ImportError("zed mode requires mediapipe, opencv-python, and pyzed")

    input_device = device_map[input_device_type]()

    # Initialize retargeter
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # qpos comes out in the URDF/Pinocchio joint order, which can differ from the
    # device's order (e.g. WH120 declares fingers index-first). Remap by joint
    # name to the MJCF joint order (== the device's finger1..5 indexing) so the
    # commanded angles land on the right joints. Identity when orders match (WH110)
    # or when the config sets no mjcf_path.
    mjcf_path = resolve_mjcf_path(config_file)
    _qpos_perm = qpos_reorder_perm(
        retargeter.optimizer.robot.dof_joint_names,
        mjcf_joint_order(mjcf_path),
    )
    # A declared optimizer.mjcf_path means a custom hand whose joint order MUST be
    # remapped; if the names can't be aligned, qpos_reorder_perm returns None — the
    # SAME value as the legitimate WH110 "no mjcf_path" case. Driving the hand on
    # that ambiguous None would silently move the wrong joints, so fail loudly here.
    # (WH110 has no mjcf_path -> mjcf_path is None -> this is skipped.)
    if mjcf_path is not None and _qpos_perm is None:
        raise ValueError(
            "config declares optimizer.mjcf_path but the URDF<->MJCF joint names "
            "could not be aligned; refusing to drive the hand with an unverified "
            "joint order (would move the wrong joints). Check optimizer.link_naming "
            "and that urdf_path and mjcf_path describe the same hand.\n"
            f"  mjcf: {mjcf_path}"
        )
    if _qpos_perm is not None:
        print(f"qpos remap active: URDF order -> MJCF/device order ({_qpos_perm.tolist()})")
    else:
        print("qpos remap: identity (no optimizer.mjcf_path; WH110-style hand)")

    # Disable recording when using replay mode
    if input_device_type == "mediapipe_replay" and enable_recording:
        print("Note: Recording disabled in replay mode")
        enable_recording = False

    # Prepare recording
    input_data_log = [] if enable_recording else None
    start_time = time.time()

    try:
        # Energize the hand LAST — after every fallible setup step above — and INSIDE
        # this try, so the finally always runs close() (and cleans the input device)
        # even if bring-up itself fails.
        if hand_model == "wh120":
            backend = Wh120Backend(
                ip=wh120_ip,
                kp=kp,
                kd=kd,
                current_limit=current_limit,
                handedness=hand_side,
            )
        else:
            backend = Wh110Backend(hand_serial=hand_serial)

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

            # Skip until the first valid frame arrives from the input device.
            if fingers_pose is None or np.allclose(fingers_pose, 0):
                time.sleep(0.01)
                continue

            # Record raw input data if enabled
            if enable_recording:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": (
                        None
                        if fingers_data["left_fingers"] is None
                        else fingers_data["left_fingers"].copy()
                    ),
                    "right_fingers": (
                        None
                        if fingers_data["right_fingers"] is None
                        else fingers_data["right_fingers"].copy()
                    ),
                })

            # Retarget to joint angles
            qpos = retargeter.retarget(fingers_pose)  # (20,)

            # FPS counter
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}")

            # Send to hardware via backend (WH110 or WH120, decided at init).
            backend.send(qpos if _qpos_perm is None else qpos[_qpos_perm])


    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        # SIGINT-safe cleanup: close backend + input device, each None-guarded (a
        # failure during setup or bring-up can leave either unbound).
        if backend is not None:
            try:
                backend.close()
            except Exception as _e:
                print(f"backend.close() raised: {type(_e).__name__}: {_e}")
        if input_device is not None:
            for method_name in ("stop", "cleanup", "close"):
                method = getattr(input_device, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

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

  # MP4 video input with MediaPipe hand detection
  python teleop_real.py --video data/right.mp4 --hand right
  python teleop_real.py --video data/right.mp4 --hand right --show-video

  # RealSense camera input with MediaPipe hand detection
  python teleop_real.py --realsense --hand right

  # ZED camera input with MediaPipe hand detection
  python teleop_real.py --zed --hand right

  # Live VisionPro input
  python teleop_real.py --input visionpro --ip <your-vision-pro-ip>

  # Record input data while using VisionPro
  python teleop_real.py --input visionpro --record

  # Compatibility tuning shortcut
  python teleop_real.py --play data/avp1.pkl --hand right --tuning
        """
    )

    # Config
    parser.add_argument('--config', type=str, default='config/adaptive_analytical_avp.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--hand', type=str, default='right', choices=['left', 'right'],
                        help='Hand side (default: right)')

    # Input device options
    parser.add_argument('--input', type=str, default=None,
                        choices=['visionpro', 'mediapipe_replay', 'video', 'realsense', 'zed', 'wuji_glove'],
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
    parser.add_argument('--video', type=str, default=None, metavar='FILE',
                        help='Play MP4 video file with MediaPipe hand detection (shortcut for --input video)')
    parser.add_argument('--realsense', action='store_true',
                        help='Use RealSense camera with MediaPipe hand detection (shortcut for --input realsense)')
    parser.add_argument('--zed', action='store_true',
                        help='Use ZED camera with MediaPipe hand detection (shortcut for --input zed)')
    parser.add_argument('--show-video', action='store_true',
                        help='Show video with MediaPipe landmarks overlay (video/realsense/zed mode)')
    parser.add_argument('--tuning', action='store_true',
                        help='Launch tuning visualization mode (deprecated; use tuning_tool.py directly)')
    parser.add_argument('--viz-config', type=str, default=None,
                        help='Path to tuning visualization config (default: config/tuning_viz.yaml)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Playback FPS for tuning mode (default: 30)')
    parser.add_argument('--device-name', type=str, default='glove',
                        help='wuji_sdk device name for Wuji Glove (default: glove)')
    parser.add_argument('--glove-sn', type=str, default='',
                        help='Wuji Glove serial number (required when multiple Wuji devices online)')

    # Hardware backend selection + WH120 connection params.
    parser.add_argument('--hand-model', type=str, default=None, choices=['wh110', 'wh120'],
                        help='Hand hardware model: wh110 (USB via wujihandpy) or wh120 (networked via wuji_sdk). '
                             'Default: inferred from the config (WH120 if it overrides the hand to a WH120 model, else WH110).')
    parser.add_argument('--hand-serial', type=str, default='',
                        help='WH110 hand serial number (for wujihandpy.Hand selection)')
    parser.add_argument('--wh120-ip', type=str, default='',
                        help='WH120 SDK address, e.g. 192.168.1.111:50001 (run a scan to find it)')
    parser.add_argument('--kp', type=float, default=3.0, help='WH120 MIT kp (default: 3.0)')
    parser.add_argument('--kd', type=float, default=0.1, help='WH120 MIT kd (default: 0.1)')
    parser.add_argument('--current-limit', type=float, default=1.5,
                        help='WH120 per-joint system current limit in amps (SDK sys_current_limit, default: 1.5)')
    # NOTE: per-joint calibration (sign/offset) is deliberately not exposed here —
    # an unvalidated sign flip / offset goes straight to the MIT controller on real
    # hardware (and can't be previewed in sim), which is unsafe as a user knob.

    args = parser.parse_args()

    # Determine input device type and paths
    input_device_type = args.input
    mediapipe_replay_path = ""
    video_path = ""

    if args.zed:
        input_device_type = "zed"
    elif args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    # Default to mediapipe_replay with data/avp1.pkl if no input specified
    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    # Auto-switch config for non-AVP input devices. For the Wuji Glove, pick the
    # WH120 config when --hand-model wh120 so the IK hand (urdf_path/mjcf_path in
    # that config) matches the physical hand being driven.
    if args.config == 'config/adaptive_analytical_avp.yaml':
        if input_device_type in ("realsense", "video", "zed"):
            args.config = 'config/adaptive_analytical_video.yaml'
        elif input_device_type == "wuji_glove":
            suffix = "wh120_" if args.hand_model == "wh120" else ""
            args.config = f'config/adaptive_analytical_wuji_glove_{suffix}{args.hand}.yaml'

    # Resolve hand model: explicit --hand-model wins; otherwise infer from the
    # config so that passing a WH120 config (e.g. ..._wh120_right.yaml) selects
    # the WH120 network backend without also needing --hand-model wh120.
    if args.hand_model is None:
        args.hand_model = _infer_hand_model(Path(__file__).parent / args.config)
        print(f"--hand-model not given; inferred '{args.hand_model}' from {args.config}")

    # Compatibility tuning mode. Prefer invoking tuning_tool.py directly.
    if args.tuning:
        viz_config_path = args.viz_config
        if viz_config_path is None:
            default_viz = Path(__file__).parent / "config" / "tuning_viz.yaml"
            if default_viz.exists():
                viz_config_path = "config/tuning_viz.yaml"
        run_tuning_mode(
            hand_side=args.hand,
            config_path=args.config,
            input_device_type=input_device_type,
            mediapipe_replay_path=mediapipe_replay_path,
            video_path=video_path,
            show_video=args.show_video,
            viz_config_path=viz_config_path,
            fps=args.fps,
            device_name=args.device_name,
            glove_sn=args.glove_sn,
        )
        return

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
        video_path=video_path,
        show_video=args.show_video,
        device_name=args.device_name,
        glove_sn=args.glove_sn,
        hand_model=args.hand_model,
        hand_serial=args.hand_serial,
        wh120_ip=args.wh120_ip,
        kp=args.kp,
        kd=args.kd,
        current_limit=args.current_limit,
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
