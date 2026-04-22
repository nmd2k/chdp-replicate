#!/usr/bin/env python3
"""Test script to verify all environments work correctly."""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, "/Users/manhdungnguy/Workspace/demo/chdp-reproduction/src")

from environments import (
    PlatformEnv,
    GoalEnv,
    HardGoalEnv,
    CatchPointEnv,
    HardMoveEnv,
)


def test_environment(env, env_name: str, num_steps: int = 100):
    """Test an environment by running random actions."""
    print(f"\n{'='*60}")
    print(f"Testing: {env_name}")
    print(f"{'='*60}")

    # Reset
    state = env.reset()
    print(f"State space: {env.observation_space}")
    print(f"Action space - Discrete: {env.discrete_action_space}, Continuous: {env.continuous_action_space}")

    success_count = 0
    total_reward = 0

    for step in range(num_steps):
        # Random action
        discrete_action = env.discrete_action_space.sample()
        continuous_params = env.continuous_action_space.sample()

        # Step
        next_state, reward, done, info = env.step(discrete_action, continuous_params)

        total_reward += reward
        if info.get("success", False):
            success_count += 1

        if done:
            state = env.reset()
        else:
            state = next_state

    print(f"✓ Completed {num_steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Success rate: {success_count / num_steps * 100:.1f}%")
    return True


def main():
    """Test all PAMDP environments."""
    print("\n" + "="*60)
    print("CHDP Environment Test Suite")
    print("="*60)

    environments = [
        (PlatformEnv(), "Platform"),
        (GoalEnv(), "Goal"),
        (HardGoalEnv(), "Hard Goal"),
        (CatchPointEnv(), "Catch Point"),
        (HardMoveEnv(n_actuators=4), "Hard Move (n=4)"),
        (HardMoveEnv(n_actuators=6), "Hard Move (n=6)"),
        (HardMoveEnv(n_actuators=8), "Hard Move (n=8)"),
        (HardMoveEnv(n_actuators=10), "Hard Move (n=10)"),
    ]

    results = []
    for env, name in environments:
        try:
            success = test_environment(env, name)
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
