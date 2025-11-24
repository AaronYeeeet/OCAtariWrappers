import numpy as np
import torch
from ocatari_wrappers.sarfa import SarfaExplainer
from load_agent import load_agent
from hackatari import HackAtari
import matplotlib.pyplot as plt
import argparse


def test_sarfa(game, agent_path, radius=5, blur=False, save_path='sarfa_output.png', show_plot=True):
    """
    Testet SARFA mit dem gegebenen Spiel und Agenten

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

    print(f"Observation shape: {obs.shape}")  # Sollte (4, 84, 84) sein

    # Konvertiere von (channels, height, width) zu (height, width, channels)
    stacked_frames = np.transpose(obs, (1, 2, 0))
    print(f"Stacked frames shape: {stacked_frames.shape}")  # Sollte (84, 84, 4) sein

    print(f"Generiere SARFA Saliency Map (radius={radius}, blur={blur})...")
    # DIREKT den PyTorch-Agent übergeben - kein Wrapper mehr nötig! 🎉
    saliency_map = explainer.generate_explanation(
        stacked_frames=stacked_frames,
        model=agent,  # ← PyTorch-Modell wird automatisch erkannt!
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
                        default="../models/Freeway/0/ppo_binary.cleanrl_model",
                        help="Path to trained agent (default: ../models/Freeway/0/ppo_binary.cleanrl_model)")
    parser.add_argument("-r", "--radius", type=int, default=5,
                        help="Radius for perturbation (default: 5)")
    parser.add_argument("-b", "--blur", action="store_true",
                        help="Use blur instead of occlusion")
    parser.add_argument("-o", "--output", type=str, default="sarfa_output.png",
                        help="Output file path for saliency map (default: sarfa_output.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't show the plot window")

    args = parser.parse_args()

    test_sarfa(args.game, args.agent, args.radius, args.blur, args.output, not args.no_show)
