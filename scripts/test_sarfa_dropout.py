"""
Test for SarfaDropoutWrapper.
Visualizes the dropout procedure - comparing original SARFA saliency with dropout versions.
Shows which objects are dropped (most salient) and how remaining objects are rendered.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import gymnasium as gym
import cv2
from collections import deque


# Create output directory
OUTPUT_DIR = Path("sarfa_dropout_output")
OUTPUT_DIR.mkdir(exist_ok=True)


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


def test_sarfa_dropout_wrapper(game="FreewayNoFrameskip-v4", model_name="Freeway",
                                num_images=5, frames_between=30, n_dropout=2,
                                use_blur=False, normalize_remaining=True):
    """
    Tests SarfaDropoutWrapper.

    Generates comparison visualizations showing:
    1. Raw input
    2. Binary mask (all objects)
    3. SARFA Saliency (full - with all objects, intensity = saliency)
    4. SARFA Dropout (top N salient objects removed, remaining normalized)

    Args:
        game: Game environment name
        model_name: Name of the model folder
        num_images: How many images to generate
        frames_between: How many frames to skip between each image
        n_dropout: Number of most salient objects to drop
        use_blur: Use blur instead of occlusion for SARFA
        normalize_remaining: Normalize intensities of remaining objects after dropout
    """
    print("=" * 60)
    print("Test: SARFA Dropout Wrapper")
    print("=" * 60)
    print(f"Game: {game}")
    print(f"Model: {model_name}")
    print(f"Generating {num_images} images, {frames_between} frames apart")
    print(f"Dropping top {n_dropout} most salient objects")
    print(f"Perturbation method: {'Blur' if use_blur else 'Occlusion'}")
    print(f"Normalize remaining: {normalize_remaining}\n")

    try:
        from ocatari.core import OCAtari
        from ocatari_wrappers import BinaryMaskWrapper, SarfaSaliencyWrapper
        from ocatari_wrappers.sarfa_dropout_wrapper import SarfaDropoutWrapper
        from load_agent import load_agent

        # 1. Load trained model
        print("Loading trained model...")
        model_path = f"models/{model_name}/0/ppo_binary.cleanrl_model"

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

        # Raw Input
        env0 = OCAtari(game, mode="ram", hud=True)
        env0 = RawDQNWrapper(env0)
        obs0_raw, _ = env0.reset()

        # Binary Mask
        env1 = OCAtari(game, mode="ram", hud=True)
        env1 = BinaryMaskWrapper(env1)
        obs1, _ = env1.reset()

        # SARFA Saliency (full - all objects with saliency intensity)
        env2 = OCAtari(game, mode="ram", hud=True)
        env2 = SarfaSaliencyWrapper(env2, trained_model=model, use_blur=use_blur, radius=3)
        obs2, _ = env2.reset()

        # SARFA Dropout (N most salient objects removed)
        env3 = OCAtari(game, mode="ram", hud=True)
        env3 = SarfaDropoutWrapper(
            env3,
            trained_model=model,
            n_dropout=n_dropout,
            use_blur=use_blur,
            radius=3,
            normalize_remaining=normalize_remaining
        )
        obs3, _ = env3.reset()

        print(f"✓ Raw input: {obs0_raw.shape}")
        print(f"✓ Binary mask: {obs1.shape}")
        print(f"✓ SARFA saliency: {obs2.shape}")
        print(f"✓ SARFA dropout (n={n_dropout}): {obs3.shape}\n")

        # 3. Generate multiple images
        print("Running simulation and generating images...")
        saved_images = []

        for img_idx in range(num_images):
            # Run N frames with same action for all envs
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

            # Get dropped object info
            dropped_info = env3.get_dropped_object_info()
            remaining_info = env3.get_remaining_object_info()

            # Create visualization
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
            total_objects = len([o for o in env1.env.objects if o is not None and o.category != "NoObject"])
            axes[1].set_title(f'Binary Mask\n({total_objects} objects detected)', fontweight='bold', fontsize=12)
            axes[1].axis('off')
            axes[1].text(0.02, 0.98, f'Min: {img1.min()}\nMax: {img1.max()}\nUnique: {len(np.unique(img1))}',
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # SARFA Saliency (full)
            img2 = obs2[0]
            axes[2].imshow(img2, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title('SARFA Saliency (Full)\n(all objects, intensity=saliency)', fontweight='bold', fontsize=12)
            axes[2].axis('off')
            axes[2].text(0.02, 0.98, f'Min: {img2.min()}\nMax: {img2.max()}\nUnique: {len(np.unique(img2))}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # SARFA Dropout
            img3 = obs3[0]
            axes[3].imshow(img3, cmap='gray', vmin=0, vmax=255)

            # Build dropped objects text
            dropped_text = f"DROPPED (top {n_dropout}):\n"
            if dropped_info:
                for cat, score in dropped_info:
                    dropped_text += f"  • {cat}: {score:.3f}\n"
            else:
                dropped_text += "  (none)\n"

            remaining_text = f"\nREMAINING ({len(remaining_info)}):\n"
            for cat, score in sorted(remaining_info, key=lambda x: x[1], reverse=True)[:5]:
                remaining_text += f"  • {cat}: {score:.3f}\n"
            if len(remaining_info) > 5:
                remaining_text += f"  ... and {len(remaining_info) - 5} more"

            axes[3].set_title(f'SARFA Dropout (n={n_dropout})\n(top salient objects removed)',
                             fontweight='bold', fontsize=12)
            axes[3].axis('off')
            axes[3].text(0.02, 0.98, dropped_text + remaining_text,
                        transform=axes[3].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=8)

            frame_num = (img_idx + 1) * frames_between

            # Build title
            title = f'{game} - Frame {frame_num}'
            title += f" | Dropout: {n_dropout}"
            title += f" | {'Blur' if use_blur else 'Occlusion'}"
            title += f" | Norm: {normalize_remaining}"

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save image
            filename = f"sarfa_dropout_{model_name.lower()}_frame{frame_num:04d}"
            filename += f"_drop{n_dropout}"
            filename += f"_{'blur' if use_blur else 'occlude'}"
            filename += f"_{'norm' if normalize_remaining else 'raw'}"
            filename += ".png"

            output_path = OUTPUT_DIR / filename
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            saved_images.append(output_path)
            print(f"✓ Saved: {output_path.name}")

            # Print dropped objects info
            if dropped_info:
                print(f"  Dropped: {[f'{cat}({score:.3f})' for cat, score in dropped_info]}")

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


def test_different_dropout_levels(game="FreewayNoFrameskip-v4", model_name="Freeway",
                                   frames_to_run=100, dropout_levels=[1, 2, 3, 5]):
    """
    Test different dropout levels on the same frame for comparison.

    Creates a single figure showing the effect of different N values.
    """
    print("=" * 60)
    print("Test: Different Dropout Levels Comparison")
    print("=" * 60)
    print(f"Game: {game}")
    print(f"Dropout levels to compare: {dropout_levels}\n")

    try:
        from ocatari.core import OCAtari
        from ocatari_wrappers import BinaryMaskWrapper, SarfaSaliencyWrapper
        from ocatari_wrappers.sarfa_dropout_wrapper import SarfaDropoutWrapper
        from load_agent import load_agent

        # Load model
        print("Loading trained model...")
        model_path = f"models/{model_name}/0/ppo_binary.cleanrl_model"
        temp_env = OCAtari(game, mode="ram", hud=True)
        temp_env.obs_mode = "dqn"

        try:
            agent, policy = load_agent(model_path, temp_env, "cpu")
            model = agent
            print(f"✓ Model loaded\n")
        except Exception as e:
            print(f"Cannot load model: {e}")
            temp_env.close()
            return

        temp_env.close()

        # Create environments for each dropout level
        envs = {}

        # Raw
        envs['raw'] = OCAtari(game, mode="ram", hud=True)
        envs['raw'] = RawDQNWrapper(envs['raw'])

        # SARFA full
        envs['sarfa_full'] = OCAtari(game, mode="ram", hud=True)
        envs['sarfa_full'] = SarfaSaliencyWrapper(envs['sarfa_full'], trained_model=model)

        # Dropout variants
        for n in dropout_levels:
            envs[f'dropout_{n}'] = OCAtari(game, mode="ram", hud=True)
            envs[f'dropout_{n}'] = SarfaDropoutWrapper(
                envs[f'dropout_{n}'],
                trained_model=model,
                n_dropout=n
            )

        # Reset all
        observations = {}
        for key, env in envs.items():
            observations[key], _ = env.reset()

        # Run frames
        print(f"Running {frames_to_run} frames...")
        for step in range(frames_to_run):
            action = envs['raw'].action_space.sample()
            for key, env in envs.items():
                obs, _, term, trunc, _ = env.step(action)
                observations[key] = obs
                if term or trunc:
                    observations[key], _ = env.reset()

        # Create comparison figure
        n_cols = 2 + len(dropout_levels)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

        # Raw
        axes[0].imshow(observations['raw'][0], cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Raw Input', fontweight='bold')
        axes[0].axis('off')

        # SARFA full
        axes[1].imshow(observations['sarfa_full'][0], cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('SARFA Full\n(all objects)', fontweight='bold')
        axes[1].axis('off')

        # Dropout variants
        for i, n in enumerate(dropout_levels):
            key = f'dropout_{n}'
            axes[2 + i].imshow(observations[key][0], cmap='gray', vmin=0, vmax=255)

            # Get dropped info
            dropped = envs[key].get_dropped_object_info()
            dropped_cats = [cat for cat, _ in dropped]

            axes[2 + i].set_title(f'Dropout n={n}\n({", ".join(dropped_cats[:3])}{"..." if len(dropped_cats) > 3 else ""})',
                                  fontweight='bold', fontsize=10)
            axes[2 + i].axis('off')

        plt.suptitle(f'{game} - Dropout Level Comparison (Frame {frames_to_run})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        filename = f"sarfa_dropout_comparison_{model_name.lower()}.png"
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"\n✓ Saved comparison: {output_path}")

        plt.close()

        # Close all environments
        for env in envs.values():
            env.close()

        print(f"\n{'=' * 60}")
        print(f"✓ Comparison complete")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SARFA Dropout Wrapper")
    parser.add_argument("-g", "--game", type=str, default="FreewayNoFrameskip-v4",
                        help="Game environment (default: FreewayNoFrameskip-v4)")
    parser.add_argument("-m", "--model", type=str, default="Freeway",
                        help="Model name (default: Freeway)")
    parser.add_argument("-n", "--num_images", type=int, default=5,
                        help="Number of images to generate (default: 5)")
    parser.add_argument("-f", "--frames_between", type=int, default=30,
                        help="Frames between each image (default: 30)")
    parser.add_argument("-d", "--dropout", type=int, default=2,
                        help="Number of most salient objects to drop (default: 2)")
    parser.add_argument("-b", "--blur", action="store_true",
                        help="Use blur instead of occlusion (default: False)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Don't normalize remaining objects (default: normalize)")
    parser.add_argument("--compare", action="store_true",
                        help="Run dropout level comparison instead of regular test")
    parser.add_argument("--dropout-levels", type=str, default="1,2,3,5",
                        help="Comma-separated dropout levels for comparison (default: 1,2,3,5)")

    args = parser.parse_args()

    if args.compare:
        dropout_levels = [int(x) for x in args.dropout_levels.split(',')]
        test_different_dropout_levels(
            game=args.game,
            model_name=args.model,
            dropout_levels=dropout_levels
        )
    else:
        test_sarfa_dropout_wrapper(
            game=args.game,
            model_name=args.model,
            num_images=args.num_images,
            frames_between=args.frames_between,
            n_dropout=args.dropout,
            use_blur=args.blur,
            normalize_remaining=not args.no_normalize
        )

