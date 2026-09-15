"""Standard inverse-kinematics baseline for Gymnasium Pusher-v5.
The controller uses damped least-squares differential IK followed by
joint-space PD control. It first moves the fingertip behind the object and then
pushes it toward the goal. No learning is used.

"""

from __future__ import annotations
import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ARM_DOF = 7

@dataclass
class EpisodeResult:
    episode: int
    episode_return: float
    final_planar_distance: float
    success: float
    mean_fingertip_object_distance: float
    mean_control_effort: float
    execution_time_seconds: float
    steps: int


def _find_id(model, object_type, candidates):
    """Return the first MuJoCo id matching one of the supplied names."""
    for name in candidates:
        idx = mujoco.mj_name2id(model, object_type, name)
        if idx >= 0:
            return idx

    return -1

class InverseKinematicsPusher:
    """Damped-least-squares IK controller for the Pusher-v5 robot."""

    def __init__(
        self,
        env,
        damping=0.08,
        ik_gain=5.0,
        kp=8.0,
        kd=0.5,
        approach_offset=0.075,
        contact_distance=0.055,
        target_height=None,
        max_joint_step=0.20,

    ):
        self.env = env
        self.base = env.unwrapped
        self.model = self.base.model
        self.data = self.base.data
        self.damping = damping
        self.ik_gain = ik_gain
        self.kp = kp
        self.kd = kd
        self.approach_offset = approach_offset
        self.contact_distance = contact_distance
        self.target_height = target_height
        self.max_joint_step = max_joint_step
        self.tip_body_id = _find_id(self.model,mujoco.mjtObj.mjOBJ_BODY,("tips_arm", "fingertip", "distal_4"),)

        self.object_body_id = _find_id(self.model, mujoco.mjtObj.mjOBJ_BODY, ("object", "obj"))
        self.goal_body_id = _find_id(self.model, mujoco.mjtObj.mjOBJ_BODY, ("goal", "target"))
        self.object_geom_id = _find_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ("object", "obj"))
        self.goal_geom_id = _find_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ("goal", "target"))

        if self.tip_body_id < 0:
            raise RuntimeError("Could not locate the Pusher-v5 fingertip body.")

        self.action_low = np.asarray(env.action_space.low, dtype=float)
        self.action_high = np.asarray(env.action_space.high, dtype=float)

    def _body_or_geom_position(self, body_id, geom_id):
        if body_id >= 0:
            return self.data.xpos[body_id].copy()

        if geom_id >= 0:
            return self.data.geom_xpos[geom_id].copy()

        raise RuntimeError("Could not locate an object required by the controller.")

    def positions(self):
        mujoco.mj_forward(self.model, self.data)
        tip = self.data.xpos[self.tip_body_id].copy()
        obj = self._body_or_geom_position(self.object_body_id, self.object_geom_id)
        goal = self._body_or_geom_position(self.goal_body_id, self.goal_geom_id)
        return tip, obj, goal

    def desired_fingertip_position(self):
        """Choose an approach point or a pushing point in the XY plane."""
        tip, obj, goal = self.positions()
        push_direction = goal[:2] - obj[:2]
        norm = np.linalg.norm(push_direction)
        if norm < 1e-9:
            return tip
        push_direction /= norm

        behind = obj[:2] - self.approach_offset * push_direction
        if np.linalg.norm(tip[:2] - behind) > self.contact_distance:
            xy_target = behind

        else:
            # Aim slightly through the object so contact force points at the goal.
            xy_target = obj[:2] + 0.035 * push_direction

        # In Pusher-v5 the tabletop is below z=0; using a fixed positive z
        # makes the planar target unreachable. Follow the object's height.
        z_target = obj[2] if self.target_height is None else self.target_height
        return np.array([xy_target[0], xy_target[1], z_target])

    def action(self):

        target = self.desired_fingertip_position()
        tip, _, _ = self.positions()
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.tip_body_id)
        jacobian = jac_pos[:, :ARM_DOF]
        error = target - tip

        # dq = J^T (J J^T + lambda^2 I)^-1 K e
        regularizer = (self.damping**2) * np.eye(3)
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, self.ik_gain * error)
        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        q = self.data.qpos[:ARM_DOF]
        q_vel = self.data.qvel[:ARM_DOF]
        desired_q = q + dq
        control = self.kp * (desired_q - q) - self.kd * q_vel

        return np.clip(control, self.action_low, self.action_high).astype(np.float32)

def evaluate_episode(env, controller, episode, seed, success_threshold):
    env.reset(seed=seed)
    rewards, planar_distances, tip_distances, efforts = [], [], [], []
    start = time.perf_counter()
    terminated = truncated = False

    while not (terminated or truncated):
        action = controller.action()
        _, reward, terminated, truncated, _ = env.step(action)
        tip, obj, goal = controller.positions()
        rewards.append(float(reward))
        planar_distances.append(float(np.linalg.norm(obj[:2] - goal[:2])))
        tip_distances.append(float(np.linalg.norm(tip - obj)))
        efforts.append(float(np.sum(np.square(action))))

    elapsed = time.perf_counter() - start
    final_distance = planar_distances[-1]
    result = EpisodeResult(
        episode=episode,
        episode_return=sum(rewards),
        final_planar_distance=final_distance,
        success=float(final_distance <= success_threshold),
        mean_fingertip_object_distance=float(np.mean(tip_distances)),
        mean_control_effort=float(np.mean(efforts)),
        execution_time_seconds=elapsed,
        steps=len(rewards),
    )

    traces = {
        "reward": np.asarray(rewards),
        "planar_distance": np.asarray(planar_distances),
        "tip_distance": np.asarray(tip_distances),
        "control_effort": np.asarray(efforts),
    }

    return result, traces


def save_outputs(results, traces, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ik_episode_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_items = [
        ("planar_distance", "Object-to-goal planar distance (m)"),
        ("tip_distance", "Fingertip-to-object distance (m)"),
        ("control_effort", "Control effort: mean squared-action norm"),
        ("reward", "Reward per step"),
    ]

    for ax, (key, ylabel) in zip(axes.flat, plot_items):
        max_length = max(len(trace[key]) for trace in traces)
        matrix = np.full((len(traces), max_length), np.nan)
        for row, trace in enumerate(traces):
            matrix[row, : len(trace[key])] = trace[key]

        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, lw=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.set(xlabel="Simulation step", ylabel=ylabel)
        ax.grid(alpha=0.25)

    fig.suptitle("Pusher-v5 Standard IK Baseline (mean +/- 1 SD)")
    fig.savefig(output_dir / "ik_performance_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ik_performance_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    episode_numbers = [r.episode for r in results]
    axes[0].bar(episode_numbers, [r.final_planar_distance for r in results])
    axes[0].set(xlabel="Episode", ylabel="Final planar distance (m)")
    axes[1].bar(episode_numbers, [r.episode_return for r in results])
    axes[1].set(xlabel="Episode", ylabel="Evaluation return")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Held-out IK Evaluation Episodes")
    fig.savefig(output_dir / "ik_held_out_evaluation.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ik_held_out_evaluation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(results):
    fields = [
        ("Evaluation return", "episode_return"),
        ("Final planar distance (m)", "final_planar_distance"),
        ("Success rate", "success"),
        ("Fingertip-object distance (m)", "mean_fingertip_object_distance"),
        ("Control effort", "mean_control_effort"),
        ("Execution time (s)", "execution_time_seconds"),
    ]

    print("\nInverse Kinematics Evaluation Summary")
    print("-" * 63)
    print(f"{'Metric':34s} {'Mean':>12s} {'Std':>12s}")

    for label, attribute in fields:
        values = np.array([getattr(result, attribute) for result in results])
        print(f"{label:34s} {values.mean():12.4f} {values.std():12.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("ik_results"))
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = gym.make("Pusher-v5", render_mode=render_mode, max_episode_steps=300)
    controller = InverseKinematicsPusher(env)
    results, all_traces = [], []
    try:
        for episode in range(1, args.episodes + 1):
            result, traces = evaluate_episode(
                env,
                controller,
                episode=episode,
                seed=args.seed + episode - 1,
                success_threshold=args.success_threshold,
            )

            results.append(result)
            all_traces.append(traces)
            print(
                f"Episode {episode:02d}: return={result.episode_return:8.2f}, "
                f"final distance={result.final_planar_distance:.4f} m, "
                f"success={int(result.success)}"
            )
    finally:
        env.close()
    save_outputs(results, all_traces, args.output_dir)
    print_summary(results)
    print(f"\nResults saved in: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()