"""
Quadcopter MuJoCo demo  -  one command, reproducible.

    python run_demo.py                      # opens the MuJoCo viewer and flies the demo in real time
    python run_demo.py --headless           # no window: fly, print metrics, save plot + JSON
    python run_demo.py --video out.mp4      # also render a two-camera video (chase view + onboard gimbal camera)
    python run_demo.py --controller smc     # sliding-mode controller instead of the cascaded PID
    python run_demo.py --camera closeup     # viewer camera: chase | closeup | overview | onboard | free

The flight is a closed-loop, gravity-on flight: takeoff from the pad -> gimbal sweep (tilt, pan) in
hover -> 3 m square with a 90 deg turn while the gimbal tracks the red vehicle -> landing -> disarm.
Controllers, trajectory generator and allocation are the project's existing modules (unchanged).

Outputs: results/demo/demo_summary.json, results/demo/demo_flight.png (+ optional video).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np


def make_sim_and_controller(controller="pid", xml_path=None):
    import mujoco
    from controllers import CascadedPIDController, SlidingModeController
    from dynamics import QuadcopterParams
    from mujoco_sim import MuJoCoQuadcopterSim
    # MuJoCo does not model ground effect / rotor gyro; switch them off in the controller's model
    # so that controller and plant agree (same choice as experiments/run_mujoco_verification.py).
    params = QuadcopterParams(enable_ground_effect=False, enable_rotor_gyro=False)
    sim = MuJoCoQuadcopterSim(xml_path=xml_path, engine=mujoco, params=params)
    C = {"pid": CascadedPIDController, "smc": SlidingModeController}[controller]
    return sim, C(params, allocator=sim.allocator)


def save_outputs(log, summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    t = log["t"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for i, (lab, c) in enumerate(zip("xyz", ["#2a78d6", "#eb6834", "#2ca05a"])):
        ax[0, 0].plot(t, log["ref"][:, i], color=c, ls=":", lw=1.2)
        ax[0, 0].plot(t, log["x"][:, i], color=c, label=f"{lab}")
    ax[0, 0].set_ylabel("position (m)  [dotted = reference]"); ax[0, 0].legend(ncol=3)
    for i in range(4):
        ax[0, 1].plot(t, log["omega"][:, i], label=f"rotor {i + 1}")
    ax[0, 1].set_ylabel("physical rotor speed (rad/s)"); ax[0, 1].legend(ncol=4, fontsize=8)
    for i in range(4):
        ax[1, 0].plot(t, log["prop_vel"][:, i], label=f"prop {i + 1}")
    ax[1, 0].set_ylabel("rendered propeller speed (rad/s)\n(+ CCW, - CW seen from above)")
    ax[1, 0].legend(ncol=4, fontsize=8); ax[1, 0].set_xlabel("time (s)")
    ax[1, 1].plot(t, np.degrees(log["gimbal"][:, 0]), label="pan")
    ax[1, 1].plot(t, np.degrees(log["gimbal"][:, 1]), label="tilt")
    ax[1, 1].plot(t, np.degrees(log["gimbal_cmd"][:, 0]), "k:", lw=1)
    ax[1, 1].plot(t, np.degrees(log["gimbal_cmd"][:, 1]), "k:", lw=1)
    ax[1, 1].set_ylabel("gimbal angle (deg)  [dotted = command]"); ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].legend()
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("MuJoCo demo flight: closed-loop, gravity on", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "demo_flight.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_video_hook(sim, path, fps, size):
    """Returns (hook, finish).  Two panels (chase | onboard gimbal camera) + a HUD strip."""
    import mujoco
    import imageio
    from PIL import Image, ImageDraw
    w, h = size
    w, h = w - w % 2, h - h % 2          # libx264 needs even frame dimensions (HUD strip is 66 px)
    renderer = mujoco.Renderer(sim.model, h, w)
    onboard_opt = mujoco.MjvOption()
    onboard_opt.geomgroup[2] = 0          # hide the landing gear in the onboard feed
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    

    def hook(k, t, sim, row, sched=None):
        renderer.update_scene(sim.data, camera="chase_cam")
        a = renderer.render().copy()
        renderer.update_scene(sim.data, camera="surveillance_cam", scene_option=onboard_opt)
        b = renderer.render().copy()
        hud = Image.new("RGB", (2 * w, 66), (18, 20, 24))
        d = ImageDraw.Draw(hud)
        x = row["x"]
        pv = row["prop_vel"]
        d.text((8, 6), f"t = {t:5.1f} s   pos = ({x[0]:5.2f}, {x[1]:5.2f}, {x[2]:4.2f}) m   "
                       f"roll/pitch = {np.degrees(x[6]):5.1f}/{np.degrees(x[7]):5.1f} deg   yaw = {np.degrees(x[8]):6.1f} deg",
               fill=(235, 235, 235))
        d.text((8, 28), "gimbal pan/tilt = %6.1f/%6.1f deg    props (rad/s shown): %s" % (
            np.degrees(row["gimbal"][0]), np.degrees(row["gimbal"][1]),
            "  ".join(f"{v:+6.1f}" for v in pv)), fill=(150, 210, 255))
        d.text((8, 50), "left: chase camera      right: onboard gimbal camera (surveillance_cam)", fill=(170, 170, 170))
        frame = np.vstack([np.hstack([a, b]), np.asarray(hud)])
        writer.append_data(frame)

    def finish():
        writer.close()
        renderer.close()

    return hook, finish


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no viewer window (metrics + plot only)")
    ap.add_argument("--controller", choices=["pid", "smc"], default="pid")
    ap.add_argument("--camera", default="chase", choices=["chase", "closeup", "overview", "onboard", "free"])
    ap.add_argument("--video", default=None, help="write an mp4 (needs imageio + imageio-ffmpeg)")
    ap.add_argument("--video-size", default="640x360", help="per-panel size WxH (default 640x360)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed factor (1 = real time)")
    ap.add_argument("--no-hold", action="store_true", help="close the viewer when the flight ends")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "demo"))
    a = ap.parse_args()

    import demo_flight as D
    sim, ctrl = make_sim_and_controller(a.controller)
    mission, sched = D.build_demo_mission()
    print(f"MuJoCo model : {sim.xml_path}")
    print(f"Controller   : {ctrl.name}   dt = {sim.dt * 1e3:.1f} ms   mission length = {sched['end']:.1f} s")

    hooks, finishers = [], []
    viewer = None
    if not a.headless:
        import mujoco
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        if a.camera == "free":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.body_id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 140.0, -20.0
        else:
            name = {"chase": "chase_cam", "closeup": "closeup_cam", "overview": "overview_cam",
                    "onboard": "surveillance_cam"}[a.camera]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if a.camera == "onboard":
                viewer.opt.geomgroup[2] = 0      # hide the landing gear in the onboard view
        stride_v = max(1, int(round(1.0 / 60.0 / sim.dt)))

        def viewer_hook(k, t, sim_, row):
            if not viewer.is_running():
                return False
            viewer.sync()
        hooks.append((stride_v, viewer_hook))
    if a.video:
        w, h = (int(v) for v in a.video_size.lower().split("x"))
        vh, fin = make_video_hook(sim, a.video, a.fps, (w, h))
        stride_c = max(1, int(round(1.0 / a.fps / sim.dt)))
        hooks.append((stride_c, vh))
        finishers.append(fin)

    def on_step(k, t, sim_, row):
        for stride, h in hooks:
            if k % stride == 0:
                if h(k, t, sim_, row) is False:
                    return False
    log = D.run_flight(sim, ctrl, mission, sched, on_step=on_step if hooks else None,
                       pace_realtime=(viewer is not None and a.speed > 0), step_stride=1)
    for f in finishers:
        f()
    summary = D.summarize(log)
    summary["controller"] = ctrl.name
    path = save_outputs(log, summary, a.out)
    print("\nDemo flight finished.")
    for k in ["position_rmse_m", "max_altitude_m", "max_roll_pitch_deg", "final_xy_error_m", "touchdown_vertical_speed_mps",
              "prop_directions", "prop_mean_speed_rpm", "gimbal_pan_range_deg", "gimbal_tilt_range_deg",
              "anomaly_los_error_mean_deg"]:
        v = summary[k]
        print(f"  {k:32s}: {np.round(v, 4) if not isinstance(v, list) or not isinstance(v[0], str) else v}")
    print(f"Summary: {os.path.join(a.out, 'demo_summary.json')}\nPlot   : {path}")
    if a.video:
        print(f"Video  : {a.video}")
    if viewer is not None and viewer.is_running() and not a.no_hold:
        print("Close the viewer window to exit.")
        while viewer.is_running():
            import time
            time.sleep(0.1)
    if viewer is not None:
        viewer.close()
    return 0 if summary["finite"] else 1


if __name__ == "__main__":
    sys.exit(main())
