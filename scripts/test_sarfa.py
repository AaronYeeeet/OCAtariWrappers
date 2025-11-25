import numpy as np
from ocatari_wrappers.sarfa import SarfaExplainer
from load_agent import load_agent
from hackatari import HackAtari
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import torch


# Create output directory
OUTPUT_DIR = Path("sarfa_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def test_sarfa_series(game, agent_path, num_images=5, frames_between=30, radius=5, blur=False):
    """
    Generiert eine Serie von SARFA Saliency Maps über mehrere Frames.

    Args:
        game: Name des Atari-Spiels (z.B. "Freeway", "Pong", "Breakout")
        agent_path: Pfad zum trainierten Agenten
        num_images: Anzahl der zu generierenden Bilder (Standard: 5)
        frames_between: Frames zwischen jedem generierten Bild (Standard: 30)
        radius: Radius für die Perturbation (Standard: 5)
        blur: Ob Blur statt Occlusion verwendet werden soll (Standard: False)
    """
    print("=" * 60)
    print("SARFA Saliency Map Serie")
    print("=" * 60)
    print(f"Spiel: {game}")
    print(f"Agent: {agent_path}")
    print(f"Generiere {num_images} Bilder, {frames_between} Frames auseinander")
    print(f"Radius: {radius}, Blur: {blur}")
    print("=" * 60)
    print()

    # Environment erstellen
    print(f"Erstelle Environment für {game}...")
    env = HackAtari(game, [], "", obs_mode="dqn", mode="ram", hud=False, render_mode=None)

    # Agent laden
    print(f"Lade Agent von {agent_path}...")
    agent, policy = load_agent(agent_path, env, "cpu")
    print(f"✓ Agent geladen: {type(agent)}")

    # SARFA Explainer erstellen
    explainer = SarfaExplainer()

    # Environment zurücksetzen
    obs, _ = env.reset()
    print(f"✓ Observation shape: {obs.shape}\n")

    # Serie generieren
    print("Generiere SARFA Maps...")
    saved_images = []

    for img_idx in range(num_images):
        # Run N frames
        for step in range(frames_between):
            action = policy(torch.Tensor(obs).unsqueeze(0))[0]
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

        # Generiere SARFA Map für diesen Frame
        frame_num = (img_idx + 1) * frames_between
        print(f"  Frame {frame_num}: Generiere SARFA Map...")

        try:
            saliency_map = explainer.generate_explanation(
                stacked_frames=obs,
                model=agent,
                radius=radius,
                blur=blur
            )

            # Erstelle Visualisierung
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Original Frame
            img_orig = obs[0]
            axes[0].imshow(img_orig, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'{game} - Frame {frame_num}\n(Original)', fontweight='bold', fontsize=12)
            axes[0].axis('off')
            axes[0].text(0.02, 0.98, f'Min: {img_orig.min()}\nMax: {img_orig.max()}',
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # SARFA Saliency Map
            im = axes[1].imshow(saliency_map, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f'SARFA Saliency Map\n(radius={radius}, blur={blur})',
                            fontweight='bold', fontsize=12)
            axes[1].axis('off')
            axes[1].text(0.02, 0.98, f'Min: {np.min(saliency_map):.3f}\nMax: {np.max(saliency_map):.3f}',
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

            # Colorbar
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # Speichern
            output_path = OUTPUT_DIR / f"sarfa_{game.lower()}_frame{frame_num:04d}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            saved_images.append(output_path)
            print(f"    ✓ Gespeichert: {output_path}")

        except Exception as e:
            print(f"    ✗ Fehler bei Frame {frame_num}: {e}")

    env.close()

    print()
    print("=" * 60)
    print(f"✓ {len(saved_images)} Bilder erfolgreich generiert")
    print(f"✓ Gespeichert in: {OUTPUT_DIR}/")
    print("=" * 60)

    return saved_images


def test_sarfa(game, agent_path, radius=5, blur=False, save_path='sarfa_output.png', show_plot=True):
    """
    Testet SARFA mit dem gegebenen Spiel und Agenten (einzelnes Bild)

    SARFA unterstützt PyTorch-Modelle direkt ohne Wrapper!

    Args:
        game: Name des Atari-Spiels (z.B. "Freeway", "Pong", "Breakout")
        agent_path: Pfad zum trainierten Agenten
        radius: Radius für die Perturbation (Standard: 5)
        blur: Ob Blur statt Occlusion verwendet werden soll (Standard: False)
        save_path: Wo die Saliency Map gespeichert werden soll
        show_plot: Ob der Plot angezeigt werden soll (Standard: True)
    """
    # Environment erstellen
    print(f"Erstelle Environment für {game}...")
    env = HackAtari(game, [], "", obs_mode="dqn", mode="ram", hud=False, render_mode=None)

    # Agent laden
    print(f"Lade Agent von {agent_path}...")
    agent, policy = load_agent(agent_path, env, "cpu")
    print(f"Agent geladen: {type(agent)}")

    # SARFA Explainer erstellen
    explainer = SarfaExplainer()

    # Environment zurücksetzen und Observation holen
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")  # (4, 84, 84) - bereits PyTorch Format!

    print(f"Generiere SARFA Saliency Map (radius={radius}, blur={blur})...")
    # DIREKT obs übergeben - kein transpose mehr nötig! 🎉
    saliency_map = explainer.generate_explanation(
        stacked_frames=obs,  # ← Direkt (C, H, W) Format!
        model=agent,
        radius=radius,
        blur=blur
    )

    print(f"Saliency map shape: {saliency_map.shape}")
    print(f"Saliency map range: [{np.min(saliency_map):.6f}, {np.max(saliency_map):.6f}]")

    # Visualisiere das Ergebnis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Zeige originalen Frame
    axes[0].imshow(obs[0], cmap='gray')
    axes[0].set_title(f'{game} - Original Frame')
    axes[0].axis('off')

    # Zeige Saliency Map
    im = axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title(f'SARFA Saliency Map (radius={radius})')
    axes[1].axis('off')

    # Füge Colorbar hinzu
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saliency Map gespeichert als '{save_path}'")

    if show_plot:
        plt.show()
    else:
        plt.close()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARFA Saliency Test für Atari-Agenten")
    parser.add_argument("-g", "--game", type=str, default="Freeway",
                        help="Atari game name (default: Freeway)")
    parser.add_argument("-a", "--agent", type=str,
                        default="models/Freeway/0/ppo_binary.cleanrl_model",
                        help="Path to trained agent (default: models/Freeway/0/ppo_binary.cleanrl_model)")
    parser.add_argument("-r", "--radius", type=int, default=5,
                        help="Radius for perturbation (default: 5)")
    parser.add_argument("-b", "--blur", action="store_true",
                        help="Use blur instead of occlusion")
    parser.add_argument("-o", "--output", type=str, default="sarfa_output.png",
                        help="Output file path for saliency map (default: sarfa_output.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't show the plot window")
    parser.add_argument("--series", action="store_true",
                        help="Generate a series of images instead of single image")
    parser.add_argument("-n", "--num-images", type=int, default=5,
                        help="Number of images to generate in series mode (default: 5)")
    parser.add_argument("-f", "--frames-between", type=int, default=30,
                        help="Frames between each image in series mode (default: 30)")

    args = parser.parse_args()

    if args.series:
        # Serie-Modus
        test_sarfa_series(
            args.game,
            args.agent,
            num_images=args.num_images,
            frames_between=args.frames_between,
            radius=args.radius,
            blur=args.blur
        )
    else:
        # Einzelbild-Modus
        test_sarfa(args.game, args.agent, args.radius, args.blur, args.output, not args.no_show)

