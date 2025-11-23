"""
Test for SARFA (Saliency-based Randomized Feature Ablation) Wrapper.
Compares different saliency methods including SARFA.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import gymnasium as gym
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directory
OUTPUT_DIR = Path("sarfa_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def test_sarfa_wrapper(game="FreewayNoFrameskip-v4", num_images=5, frames_between=30):
    """
    Tests SARFA wrapper and compares it with other saliency methods.

    Args:
        game: Atari game to test on
        num_images: How many images to generate
        frames_between: How many frames to skip between each image
    """
    print("=" * 60)
    print("Test: SARFA Saliency Wrapper")
    print("=" * 60)
    print(f"Game: {game}")
    print(f"Generating {num_images} images, {frames_between} frames apart\n")

    try:
        from ocatari.core import OCAtari
        from ocatari_wrappers import GradientSaliencyWrapper, BinaryMaskWrapper, SARFAWrapper
        from load_agent import load_agent

        # 1. Load trained model
        print("Loading trained model...")

        # Determine model path based on game
        if "Freeway" in game:
            model_path = "../models/Freeway/0/ppo_binary.cleanrl_model"
        elif "MsPacman" in game or "Pacman" in game:
            model_path = "../models/MsPacman/0/dqn_model.pth"
        elif "Breakout" in game:
            model_path = "../models/Breakout/0/dqn_model.pth"
        else:
            print(f"Warning: No model path configured for {game}")
            model_path = None

        # Temp environment for load_agent
        temp_env = OCAtari(game, mode="ram", hud=True)
        temp_env.obs_mode = "dqn"

        model = None
        if model_path and os.path.exists(model_path):
            try:
                agent, policy = load_agent(model_path, temp_env, "cpu")
                print(f"✓ Model loaded: {model_path}\n")
                model = agent
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                print("Continuing without model...\n")
        else:
            print(f"Warning: Model file not found: {model_path}")
            print("Continuing without model...\n")

        temp_env.close()

        # 2. Create environments
        print("Creating environments...")

        # Raw DQN wrapper for comparison
        from ocatari_wrappers.masked_dqn import MaskedBaseWrapper
        import cv2
        from collections import deque

        class RawDQNWrapper(gym.ObservationWrapper):
            """Simple DQN wrapper for raw Atari frames."""
            def __init__(self, env):
                super().__init__(env)
                self._buffer = deque(maxlen=4)
                self.observation_space = gym.spaces.Box(0, 255, shape=(4, 84, 84))

            def observation(self, observation):
                state = self.unwrapped.ale.getScreenGrayscale()
                state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
                self._buffer.append(state)
                return np.asarray(self._buffer)

            def reset(self, *args, **kwargs):
                ret = super().reset(*args, **kwargs)
                obs = None
                for _ in range(4):
                    obs = self.observation(ret[0])
                return obs, *ret[1:]

        # Create environments
        env0 = OCAtari(game, mode="ram", hud=True)
        env0 = RawDQNWrapper(env0)
        obs0_raw, _ = env0.reset()

        env1 = OCAtari(game, mode="ram", hud=True)
        env1 = BinaryMaskWrapper(env1)
        obs1, _ = env1.reset()

        env2 = OCAtari(game, mode="ram", hud=True)
        env2 = GradientSaliencyWrapper(env2, trained_model=model) if model else BinaryMaskWrapper(env2)
        obs2, _ = env2.reset()

        # SARFA - Create a helper object for saliency generation (not used for playing!)
        sarfa_helper = None
        if model:
            sarfa_temp_env = OCAtari(game, mode="ram", hud=True)
            sarfa_helper = SARFAWrapper(sarfa_temp_env, trained_model=model, r=5, blur=True, density=7)
            sarfa_temp_env.close()  # We only need the wrapper object, not the env

        print(f"✓ Raw input: {obs0_raw.shape}")
        print(f"✓ Binary mask: {obs1.shape}")
        print(f"✓ Gradient saliency: {obs2.shape}")
        if sarfa_helper:
            print(f"✓ SARFA helper created\n")

        # 3. Run comparison
        print("Running game and generating saliency maps...")
        print("-" * 60)

        for img_idx in range(num_images):
            print(f"\nImage {img_idx + 1}/{num_images}")

            # Take steps
            for _ in range(frames_between):
                # Get actions (random if no model)
                if model:
                    with torch.no_grad():
                        action0, _, _, _ = model.get_action_and_value(
                            torch.FloatTensor(obs0_raw).unsqueeze(0)
                        )
                        action0 = action0.item()
                else:
                    action0 = env0.action_space.sample()

                action1 = action0
                action2 = action0

                obs0_raw, _, done0, _, _ = env0.step(action0)
                obs1, _, done1, _, _ = env1.step(action1)
                obs2, _, done2, _, _ = env2.step(action2)

                if done0:
                    obs0_raw, _ = env0.reset()
                    obs1, _ = env1.reset()
                    obs2, _ = env2.reset()

            # Generate SARFA saliency map explicitly using raw frames
            print("  Generating SARFA saliency map...")
            try:
                if sarfa_helper and model:
                    sarfa_map = sarfa_helper.generate_saliency_map(obs0_raw, action=None)
                else:
                    sarfa_map = np.zeros((84, 84))
                    print("  (No model available, using zeros)")
            except Exception as e:
                print(f"  Warning: Could not generate SARFA map: {e}")
                import traceback
                traceback.print_exc()
                sarfa_map = np.zeros((84, 84))

            # 4. Visualize comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Observations
            # Raw input (last frame)
            axes[0, 0].imshow(obs0_raw[-1], cmap='gray')
            axes[0, 0].set_title('Raw Input (DQN)')
            axes[0, 0].axis('off')

            # Binary mask
            axes[0, 1].imshow(obs1[-1], cmap='gray')
            axes[0, 1].set_title('Binary Mask')
            axes[0, 1].axis('off')

            # Gradient saliency
            if model and isinstance(env2, GradientSaliencyWrapper):
                axes[0, 2].imshow(obs2[-1], cmap='gray')
                axes[0, 2].set_title('Gradient Saliency')
            else:
                axes[0, 2].imshow(obs2[-1], cmap='gray')
                axes[0, 2].set_title('Gradient (No Model)')
            axes[0, 2].axis('off')

            # Row 2: Saliency Maps
            # Gradient saliency map
            if model and hasattr(env2, 'gradient_map') and env2.gradient_map is not None:
                axes[1, 0].imshow(env2.gradient_map, cmap='jet')
                axes[1, 0].set_title('Gradient Saliency Map')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Gradient Map', ha='center', va='center')
                axes[1, 0].set_title('Gradient Saliency Map')
            axes[1, 0].axis('off')

            # SARFA saliency map
            axes[1, 1].imshow(sarfa_map, cmap='jet')
            axes[1, 1].set_title('SARFA Saliency Map')
            axes[1, 1].axis('off')

            # SARFA overlay on raw input
            axes[1, 2].imshow(obs0_raw[-1], cmap='gray', alpha=0.6)
            axes[1, 2].imshow(sarfa_map, cmap='jet', alpha=0.4)
            axes[1, 2].set_title('SARFA Overlay')
            axes[1, 2].axis('off')

            plt.tight_layout()
            output_path = OUTPUT_DIR / f"sarfa_comparison_{game.split('-')[0]}_{img_idx}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
            plt.close()

        # Close environments
        env0.close()
        env1.close()
        env2.close()

        print("\n" + "=" * 60)
        print(f"✓ Test completed! Images saved to {OUTPUT_DIR}")
        print("=" * 60)

    except ImportError as e:
        print(f"Error: Missing required module: {e}")
        print("Make sure OCAtari and all dependencies are installed.")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def test_sarfa_parameters(game="FreewayNoFrameskip-v4"):
    """
    Test SARFA with different parameters (radius, blur, density).

    Args:
        game: Atari game to test on
    """
    print("\n" + "=" * 60)
    print("Test: SARFA Parameter Comparison")
    print("=" * 60)

    try:
        from ocatari.core import OCAtari
        from ocatari_wrappers import SARFAWrapper
        from load_agent import load_agent
        import cv2

        # Load model
        print("Loading model...")
        if "Freeway" in game:
            model_path = "../models/Freeway/0/ppo_binary.cleanrl_model"
        else:
            model_path = None

        temp_env = OCAtari(game, mode="ram", hud=True)
        temp_env.obs_mode = "dqn"

        model = None
        if model_path and os.path.exists(model_path):
            try:
                agent, policy = load_agent(model_path, temp_env, "cpu")
                model = agent
                print(f"✓ Model loaded\n")
            except Exception as e:
                print(f"Warning: Could not load model: {e}\n")

        temp_env.close()

        if not model:
            print("Cannot run parameter comparison without model. Skipping...")
            return

        # Create base environment
        env = OCAtari(game, mode="ram", hud=True)
        env.reset()

        # Get a frame
        for _ in range(30):
            action = env.action_space.sample()
            env.step(action)

        # Get current state
        state = env.unwrapped.ale.getScreenGrayscale()
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
        stacked_state = np.stack([state] * 4, axis=0)

        # Test different parameters
        configs = [
            {"r": 3, "blur": True, "density": 5, "name": "r=3, blur"},
            {"r": 5, "blur": True, "density": 5, "name": "r=5, blur"},
            {"r": 7, "blur": True, "density": 5, "name": "r=7, blur"},
            {"r": 5, "blur": False, "density": 5, "name": "r=5, occlude"},
            {"r": 5, "blur": True, "density": 3, "name": "r=5, d=3"},
            {"r": 5, "blur": True, "density": 7, "name": "r=5, d=7"},
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, config in enumerate(configs):
            print(f"Testing config: {config['name']}")

            # Create wrapper with specific parameters
            test_env = OCAtari(game, mode="ram", hud=True)
            test_env = SARFAWrapper(
                test_env,
                trained_model=model,
                r=config["r"],
                blur=config["blur"],
                density=config["density"]
            )
            test_env.reset()

            # Generate saliency map
            try:
                saliency_map = test_env.generate_saliency_map(stacked_state)
                axes[idx].imshow(saliency_map, cmap='jet')
                axes[idx].set_title(config['name'])
            except Exception as e:
                print(f"  Error: {e}")
                axes[idx].text(0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                axes[idx].set_title(config['name'])

            axes[idx].axis('off')
            test_env.close()

        plt.tight_layout()
        output_path = OUTPUT_DIR / f"sarfa_parameters_{game.split('-')[0]}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Parameter comparison saved: {output_path}")
        plt.close()

        env.close()

    except Exception as e:
        print(f"Error during parameter test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run tests
    print("Starting SARFA Wrapper Tests\n")

    # Test 1: Basic SARFA wrapper comparison
    test_sarfa_wrapper(game="FreewayNoFrameskip-v4", num_images=3, frames_between=30)

    # Test 2: Parameter comparison
    # test_sarfa_parameters(game="FreewayNoFrameskip-v4")

    print("\n✓ All tests completed!")

