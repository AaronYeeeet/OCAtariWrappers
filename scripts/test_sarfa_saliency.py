"""
Test for SarfaSaliencyWrapper.
Compares Binary Mask, Gradient Saliency, and SARFA Saliency.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import gymnasium as gym


# Create output directory (in parent OCAtariWrappers folder)
OUTPUT_DIR = Path("sarfa_saliency_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def test_sarfa_saliency_wrapper(game="FreewayNoFrameskip-v4", model_name="Freeway",
                                 num_images=5, frames_between=30, use_blur=False, use_masked=False):
    """
    Tests SarfaSaliencyWrapper.

    Args:
        game: Game environment name
        model_name: Name of the model folder
        num_images: How many images to generate
        frames_between: How many frames to skip between each image
        use_blur: Use blur instead of occlusion for SARFA
        use_masked: Use binary masked frames instead of raw grayscale
    """
    print("=" * 60)
    print("Test: SARFA Saliency Wrapper")
    print("=" * 60)
    print(f"Game: {game}")
    print(f"Model: {model_name}")
    print(f"Generating {num_images} images, {frames_between} frames apart")
    print(f"Perturbation method: {'Blur' if use_blur else 'Occlusion'}")
    print(f"Frame mode: {'Binary Masked' if use_masked else 'Raw Grayscale'}\n")

    try:
        from ocatari.core import OCAtari
        from ocatari_wrappers import BinaryMaskWrapper, GradientSaliencyWrapper, SarfaSaliencyWrapper
        from load_agent import load_agent

        # 1. Load trained model
        print("Loading trained model...")
        model_path = f"models/{model_name}/0/ppo_binary.cleanrl_model"

        # Temp environment for load_agent
        temp_env = OCAtari(game, mode="ram", hud=True)
        temp_env.obs_mode = "dqn"

        try:
            agent, policy = load_agent(model_path, temp_env, "cpu")
            print(f"✓ Model loaded: {model_path}\n")
            model = agent
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Cannot run SARFA without model. Exiting.\n")
            temp_env.close()
            return

        temp_env.close()

        # 2. Create environments
        print("Creating environments...")

        # Raw Input (DQN-style grayscale)
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

        # SARFA Saliency
        env3 = OCAtari(game, mode="ram", hud=True)
        env3 = SarfaSaliencyWrapper(env3, trained_model=model, use_blur=use_blur, radius=3, use_binary_mask=use_masked)
        obs3, _ = env3.reset()

        print(f"✓ Raw input: {obs0_raw.shape}")
        print(f"✓ Binary mask: {obs1.shape}")
        print(f"✓ Gradient saliency: {obs2.shape}")
        print(f"✓ SARFA saliency: {obs3.shape}\n")

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
                obs3, _, term3, trunc3, _ = env3.step(action)

                if term0 or trunc0:
                    obs0_raw, _ = env0.reset()
                if term1 or trunc1:
                    obs1, _ = env1.reset()
                if term2 or trunc2:
                    obs2, _ = env2.reset()
                if term3 or trunc3:
                    obs3, _ = env3.reset()

            # Create visualization for this frame
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            axes = axes.flatten()

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
            axes[2].set_title('Gradient Saliency\n(gradient-based)', fontweight='bold', fontsize=12)
            axes[2].axis('off')
            axes[2].text(0.02, 0.98, f'Min: {img2.min()}\nMax: {img2.max()}\nUnique: {len(np.unique(img2))}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # SARFA Saliency
            img3 = obs3[0]
            axes[3].imshow(img3, cmap='gray', vmin=0, vmax=255)
            axes[3].set_title(f'SARFA Saliency\n({"blur" if use_blur else "occlusion"}-based)', fontweight='bold', fontsize=12)
            axes[3].axis('off')
            axes[3].text(0.02, 0.98, f'Min: {img3.min()}\nMax: {img3.max()}\nUnique: {len(np.unique(img3))}',
                        transform=axes[3].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            frame_num = (img_idx + 1) * frames_between

            # Build descriptive title
            title = f'{game} - Frame {frame_num}'
            title += f" | {'Masked' if use_masked else 'Raw'}"
            title += f" | {'Blur' if use_blur else 'Occlusion'}"

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Build descriptive filename with flag indicators
            filename = f"sarfa_{model_name.lower()}_frame{frame_num:04d}"
            filename += f"_{'masked' if use_masked else 'raw'}"
            filename += f"_{'blur' if use_blur else 'occlude'}"
            filename += ".png"

            output_path = OUTPUT_DIR / filename
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            saved_images.append(output_path)
            print(f"✓ Saved: {output_path.name}")

            plt.close()

        # Close environments
        env0.close()
        env1.close()
        env2.close()
        env3.close()

        print(f"\n{'=' * 60}")
        print(f"✓ Successfully generated {len(saved_images)} images")
        print(f"✓ Output directory: {OUTPUT_DIR}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SARFA Saliency Wrapper")
    parser.add_argument("-g", "--game", type=str, default="FreewayNoFrameskip-v4",
                        help="Game environment (default: FreewayNoFrameskip-v4)")
    parser.add_argument("-m", "--model", type=str, default="Freeway",
                        help="Model name (default: Freeway)")
    parser.add_argument("-n", "--num_images", type=int, default=5,
                        help="Number of images to generate (default: 5)")
    parser.add_argument("-f", "--frames_between", type=int, default=30,
                        help="Frames between each image (default: 30)")
    parser.add_argument("-b", "--blur", action="store_true",
                        help="Use blur instead of occlusion (default: False)")
    parser.add_argument("--use-masked", action="store_true",
                        help="Use binary masked frames instead of raw grayscale (default: False)")

    args = parser.parse_args()

    test_sarfa_saliency_wrapper(
        game=args.game,
        model_name=args.model,
        num_images=args.num_images,
        frames_between=args.frames_between,
        use_blur=args.blur,
        use_masked=args.use_masked
    )

