"""HUD (Heads-Up Display) text overlay for MuJoCo viewer.

Displays parameter values, frame info, and optimization cost
as text overlay in the viewer window.
"""

import numpy as np


def format_hud_text(
    frame_idx: int = -1,
    total_frames: int = -1,
    cost: float = None,
    pinch_alphas: np.ndarray = None,
    retarget_config: dict = None,
    show_params: bool = True,
    show_frame_info: bool = True,
    show_cost: bool = True,
    show_pinch_alpha: bool = True,
    paused: bool = False,
) -> str:
    """Format HUD text string for overlay display.

    Args:
        frame_idx: Current frame index
        total_frames: Total number of frames
        cost: Optimization cost value
        pinch_alphas: (5,) per-finger pinch alpha values
        retarget_config: The 'retarget' section of the config dict
        show_params: Show key parameter values
        show_frame_info: Show frame counter
        show_cost: Show optimization cost
        show_pinch_alpha: Show pinch alpha values
        paused: Whether playback is paused

    Returns:
        Multi-line string for HUD display
    """
    lines = []
    lines.append("=== Tuning Viewer ===")

    if show_frame_info and frame_idx >= 0:
        status = " [PAUSED]" if paused else ""
        if total_frames > 0:
            lines.append(f"Frame: {frame_idx}/{total_frames}{status}")
        else:
            lines.append(f"Frame: {frame_idx}{status}")

    if show_cost and cost is not None:
        lines.append(f"Cost: {cost:.4f}")

    if show_pinch_alpha and pinch_alphas is not None:
        finger_names = ["Th", "Ix", "Md", "Rg", "Pk"]
        alpha_str = " ".join(
            f"{n}:{a:.2f}" for n, a in zip(finger_names, pinch_alphas)
        )
        lines.append(f"Pinch: {alpha_str}")

    if show_params and retarget_config:
        lines.append("--- Parameters ---")
        # Show key tuning parameters
        scaling = retarget_config.get("scaling", 1.0)
        lines.append(f"scaling: {scaling}")

        lp_alpha = retarget_config.get("lp_alpha", 0.2)
        lines.append(f"lp_alpha: {lp_alpha}")

        norm_delta = retarget_config.get("norm_delta", 0.04)
        lines.append(f"norm_delta: {norm_delta}")

        seg = retarget_config.get("segment_scaling", {})
        if seg:
            lines.append("segment_scaling:")
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                vals = seg.get(finger, [1.0, 1.0, 1.0])
                vals_str = ", ".join(f"{v:.2f}" for v in vals)
                lines.append(f"  {finger}: [{vals_str}]")

    lines.append("")
    lines.append("Keys: SPACE=pause  <-/-> =frame  R=reload  Q=quit")

    return "\n".join(lines)
