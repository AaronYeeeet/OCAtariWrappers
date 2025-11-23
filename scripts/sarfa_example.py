"""
Einfaches Beispiel zur Verwendung des SARFA Wrappers.
Generiert eine SARFA Saliency Map für eine einzelne Observation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocatari.core import OCAtari
from ocatari_wrappers import SARFAWrapper


def simple_sarfa_example():
    """
    Einfaches Beispiel: SARFA Saliency Map für Ms. Pac-Man generieren.
    """
    print("=" * 60)
    print("SARFA Wrapper - Einfaches Beispiel")
    print("=" * 60)

    # 1. Umgebung erstellen
    print("\n1. Erstelle Umgebung...")
    game = "MsPacman-v5"
    env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")

    # 2. Modell laden (optional - kann auch None sein für Test)
    print("2. Lade Modell...")
    model_path = "../models/MsPacman/0/dqn_model.pth"

    model = None
    if os.path.exists(model_path):
        print(f"   Hinweis: Modell gefunden, aber für dieses Beispiel verwenden wir None")
        print(f"   (Für echte Saliency Maps, lade das Modell mit load_agent)")
    else:
        print(f"   Hinweis: Kein Modell gefunden unter {model_path}")
        print(f"   SARFA benötigt ein trainiertes Modell für echte Ergebnisse")

    # 3. SARFA Wrapper anwenden
    print("\n3. Erstelle SARFA Wrapper...")
    # Für Demo-Zwecke ohne Modell (würde normalerweise model=agent verwenden)
    wrapped_env = SARFAWrapper(
        env,
        trained_model=None,  # Ersetze mit echtem Modell für echte Saliency Maps
        r=5,                  # Perturbations-Radius
        blur=True,           # Gaussian Blur statt Okklusion
        density=5            # Sampling-Dichte
    )

    # 4. Environment zurücksetzen und einige Steps ausführen
    print("4. Spiele einige Frames...")
    obs, _ = wrapped_env.reset()

    # Führe einige zufällige Aktionen aus
    for i in range(30):
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        if done:
            obs, _ = wrapped_env.reset()

    # 5. Zeige aktuelle Observation
    print("5. Zeige Observation...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Zeige letzte Frame
    axes[0].imshow(obs[-1], cmap='gray')
    axes[0].set_title('Aktuelle Observation (letzte Frame)')
    axes[0].axis('off')

    # Zeige Objekte (falls OCAtari sie erkannt hat)
    if hasattr(env, 'objects') and len(env.objects) > 0:
        info_text = f"Erkannte Objekte: {len([o for o in env.objects if o is not None])}"
        axes[0].text(0.5, -0.1, info_text, ha='center', transform=axes[0].transAxes)

    # Info-Text für SARFA
    info_lines = [
        "SARFA Wrapper erstellt!",
        "",
        "Um echte Saliency Maps zu generieren:",
        "1. Lade ein trainiertes Modell",
        "2. Rufe env.generate_saliency_map(obs, action) auf",
        "",
        "Beispiel:",
        "  saliency = env.generate_saliency_map(obs, action=2)",
    ]

    axes[1].text(0.1, 0.5, '\n'.join(info_lines),
                 fontsize=10, family='monospace',
                 verticalalignment='center')
    axes[1].axis('off')

    plt.tight_layout()

    # Speichere Bild
    output_dir = Path("sarfa_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sarfa_simple_example.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Beispiel-Bild gespeichert: {output_path}")
    plt.show()

    # Cleanup
    wrapped_env.close()

    print("\n" + "=" * 60)
    print("Beispiel abgeschlossen!")
    print("=" * 60)
    print("\nNächste Schritte:")
    print("  1. Lade ein trainiertes Modell")
    print("  2. Verwende generate_saliency_map() für echte Saliency Maps")
    print("  3. Siehe test_sarfa.py für vollständiges Beispiel")
    print("  4. Siehe SARFA_README.md für detaillierte Dokumentation")


def sarfa_with_model_example():
    """
    Beispiel mit trainiertem Modell (wenn verfügbar).
    """
    print("\n" + "=" * 60)
    print("SARFA mit Modell - Beispiel")
    print("=" * 60)

    try:
        from load_agent import load_agent
        import torch

        # Setup
        game = "FreewayNoFrameskip-v4"
        model_path = "../models/Freeway/0/ppo_binary.cleanrl_model"

        if not os.path.exists(model_path):
            print(f"Modell nicht gefunden: {model_path}")
            print("Überspringe Beispiel mit Modell...")
            return

        print(f"\n1. Lade Modell: {model_path}")

        # Temp environment für load_agent
        temp_env = OCAtari(game, mode="ram", hud=True)
        temp_env.obs_mode = "dqn"

        agent, policy = load_agent(model_path, temp_env, "cpu")
        print("   ✓ Modell geladen")
        temp_env.close()

        # Create SARFA environment
        print("\n2. Erstelle SARFA Environment...")
        env = OCAtari(game, mode="ram", hud=True)
        env = SARFAWrapper(env, trained_model=agent, r=5, blur=True, density=5)

        # Reset
        obs, _ = env.reset()

        # Play some frames
        print("3. Spiele einige Frames...")
        for _ in range(30):
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    torch.FloatTensor(obs).unsqueeze(0)
                )
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)
            if done:
                obs, _ = env.reset()

        # Generate SARFA saliency map
        print("4. Generiere SARFA Saliency Map...")
        print("   (Dies kann einige Sekunden dauern...)")
        saliency_map = env.generate_saliency_map(obs, action=None)

        # Visualize
        print("5. Visualisiere Ergebnis...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original observation
        axes[0].imshow(obs[-1], cmap='gray')
        axes[0].set_title('Observation')
        axes[0].axis('off')

        # SARFA saliency map
        axes[1].imshow(saliency_map, cmap='jet')
        axes[1].set_title('SARFA Saliency Map')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(obs[-1], cmap='gray', alpha=0.6)
        axes[2].imshow(saliency_map, cmap='jet', alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        # Save
        output_dir = Path("sarfa_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "sarfa_with_model_example.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Gespeichert: {output_path}")
        plt.show()

        env.close()

        print("\n✓ Beispiel mit Modell abgeschlossen!")

    except ImportError as e:
        print(f"\nFehler: {e}")
        print("Stelle sicher, dass alle Dependencies installiert sind.")
    except Exception as e:
        print(f"\nFehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Führe einfaches Beispiel aus (ohne Modell)
    simple_sarfa_example()

    # Führe Beispiel mit Modell aus (wenn verfügbar)
    # sarfa_with_model_example()

