"""
Test for GradientSaliencyWrapper.
Compares Binary Mask with Gradient Saliency.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import gymnasium as gym


# Create output directory
OUTPUT_DIR = Path("gradient_saliency_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def test_gradient_saliency_wrapper(num_images=5, frames_between=30):
    """
    Tests GradientSaliencyWrapper.

    Args:
        num_images: How many images to generate
        frames_between: How many frames to skip between each image
    """
    print("=" * 60)
    print("Test: Gradient Saliency Wrapper")
    print("=" * 60)
    print(f"Generating {num_images} images, {frames_between} frames apart\n")

    try:
        from ocatari.core import OCAtari
        from ocatari_wrappers import GradientSaliencyWrapper, BinaryMaskWrapper
        from load_agent import load_agent

        game = "FreewayNoFrameskip-v4"

        # 1. Load trained model
        print("Loading trained model...")
        model_path = "../models/Freeway/0/ppo_binary.cleanrl_model"

        # Temp environment for load_agent
        temp_env = OCAtari(game, mode="ram", hud=True)
        temp_env.obs_mode = "dqn"

        try:
            agent, policy = load_agent(model_path, temp_env, "cpu")
            print(f"✓ Model loaded: {model_path}\n")
            model = agent
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Using mock model...\n")
            model = None

        temp_env.close()

        # 2. Create environments
        print("Creating environments...")

        # Raw Input (DQN-style grayscale - original Atari screen)
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
                for _ in range(4):
                    obs = self.observation(ret[0])
                return obs, *ret[1:]

        env0 = OCAtari(game, mode="ram", hud=True)
        env0 = RawDQNWrapper(env0)
        obs0_raw, _ = env0.reset()

        # Binary Mask
        env1 = OCAtari(game, mode="ram", hud=True)
        env1 = BinaryMaskWrapper(env1)
        obs1, _ = env1.reset()

        # Gradient Saliency
        env2 = OCAtari(game, mode="ram", hud=True)
        env2 = GradientSaliencyWrapper(env2, trained_model=model)
        obs2, _ = env2.reset()

        print(f"✓ Raw input: {obs0_raw.shape}")
        print(f"✓ Binary mask: {obs1.shape}")
        print(f"✓ Gradient saliency: {obs2.shape}\n")

        # 3. Generate multiple images
        print("Running simulation and generating images...")
        saved_images = []

        for img_idx in range(num_images):
            # Run N frames
            for step in range(frames_between):
                action = env0.action_space.sample()
                obs0_raw, _, term0, trunc0, _ = env0.step(action)
                obs1, _, term1, trunc1, _ = env1.step(action)
                obs2, _, term2, trunc2, _ = env2.step(action)

                if term0 or trunc0:
                    obs0_raw, _ = env0.reset()
                if term1 or trunc1:
                    obs1, _ = env1.reset()
                if term2 or trunc2:
                    obs2, _ = env2.reset()

            # Create visualization for this frame
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Raw Input
            img0 = obs0_raw[0]
            axes[0].imshow(img0, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title('Raw Input\n(original screen)', fontweight='bold', fontsize=12)
            axes[0].axis('off')
            axes[0].text(0.02, 0.98, f'Min: {img0.min()}\nMax: {img0.max()}\nUnique: {len(np.unique(img0))}',
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # Binary Mask
            img1 = obs1[0]
            axes[1].imshow(img1, cmap='gray', vmin=0, vmax=255)
            axes[1].set_title('Binary Mask\n(all objects equal)', fontweight='bold', fontsize=12)
            axes[1].axis('off')
            axes[1].text(0.02, 0.98, f'Min: {img1.min()}\nMax: {img1.max()}\nUnique: {len(np.unique(img1))}',
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # Gradient Saliency
            img2 = obs2[0]
            axes[2].imshow(img2, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title('Gradient Saliency\n(weighted)', fontweight='bold', fontsize=12)
            axes[2].axis('off')
            axes[2].text(0.02, 0.98, f'Min: {img2.min()}\nMax: {img2.max()}\nUnique: {len(np.unique(img2))}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            frame_num = (img_idx + 1) * frames_between
            plt.suptitle(f'Frame {frame_num}: Raw Input vs. Binary vs. Gradient Saliency ({game})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save
            output_path = OUTPUT_DIR / f"comparison_frame_{frame_num:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Image {img_idx + 1}/{num_images} saved: {output_path.name}")
            plt.close()

            saved_images.append(output_path)

        print(f"\n✓ Generated {len(saved_images)} images\n")

        # Cleanup
        env0.close()
        env1.close()
        env2.close()

        print("=" * 60)
        print("✅ Test passed")
        print("=" * 60)
        print(f"\nResult: {OUTPUT_DIR.absolute()}")
        print(f"Generated {len(saved_images)} images:")
        for img_path in saved_images:
            print(f"  - {img_path.name}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gradient_saliency_wrapper()
    exit(0 if success else 1)

