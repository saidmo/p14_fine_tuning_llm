"""Démarrage du serveur vLLM sur le pod GPU (semaine 4).

À lancer DANS le pod (image vllm/vllm-openai), depuis la racine du dépôt :
    python -m src.serving.demarrer_vllm              # contrôles, téléchargement, puis vllm serve
    python -m src.serving.demarrer_vllm --verifier   # contrôles et commande affichée, sans lancer

Fermé par défaut : le serveur ne démarre pas si
  - l'image n'est pas épinglée dans configs/serving.yaml, ou si la version de vLLM installée
    n'est pas celle de l'étiquette (le pod ne tourne pas sur l'image prévue) ;
  - HF_TOKEN (dépôts privés) ou VLLM_API_KEY (protection de l'API) est absent.

Le modèle de base est chargé par vLLM à sa révision épinglée (--revision). L'adaptateur DPO est
téléchargé à SA révision épinglée dans un dossier propre, puis passé par son chemin : il est
appliqué à l'exécution, jamais fusionné. La clé d'API n'est jamais passée en argument (elle
serait visible dans la liste des processus) : vLLM la lit dans VLLM_API_KEY.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path, PurePath

from src.api.config import ConfigServing, charger_config


def verifier_environnement(config: ConfigServing) -> None:
    config.vllm.verifier_image()                                   
    import vllm                                                    # présent dans l'image vLLM seulement
    installee = vllm.__version__.split("+")[0]           # « 0.31.0+cu129 » → « 0.31.0 »
    if installee != config.vllm.version_vllm:
        raise SystemExit(f"vLLM installé {vllm.__version__} ≠ version épinglée "
                         f"{config.vllm.version_vllm} ({config.vllm.image}) : "
                         "le pod ne tourne pas sur l'image prévue")
    manquantes = [v for v in ("HF_TOKEN", "VLLM_API_KEY") if not os.environ.get(v)]
    if manquantes:
        raise SystemExit(f"variables d'environnement manquantes : {', '.join(manquantes)}")


def telecharger_adaptateurs(config: ConfigServing) -> dict[str, Path]:
    """Un dossier par révision : changer de révision ne réutilise jamais d'anciens fichiers."""
    from huggingface_hub import snapshot_download
    chemins = {}
    for a in config.vllm.adaptateurs:
        dossier = Path(config.vllm.dossier_modeles) / a.nom_servi / a.revision
        snapshot_download(a.repo, revision=a.revision, local_dir=dossier)
        chemins[a.nom_servi] = dossier
    return chemins


def commande(config: ConfigServing, chemins: dict[str, PurePath]) -> list[str]:
    """Commande vllm serve, entièrement déduite de configs/serving.yaml."""
    v = config.vllm
    cmd = ["vllm", "serve", v.modele.repo,
           "--revision", v.modele.revision,
           "--tokenizer-revision", v.modele.revision,
           "--served-model-name", v.modele.nom_servi,
           "--download-dir", v.dossier_modeles,                     # volume : pas de retéléchargement
           "--dtype", v.dtype,
           "--max-model-len", str(v.max_model_len),
           "--gpu-memory-utilization", str(v.gpu_memory_utilization),
           "--generation-config", "vllm",       # ignorer les valeurs par défaut du dépôt du modèle :
                                                # la passerelle envoie tous les paramètres explicitement
           "--host", "0.0.0.0", "--port", str(v.port)]
    if v.adaptateurs:
        cmd += ["--enable-lora",
                "--max-lora-rank", str(max(a.rang for a in v.adaptateurs)),
                "--max-loras", str(len(v.adaptateurs)),
                "--lora-modules", *[f"{a.nom_servi}={chemins[a.nom_servi]}" for a in v.adaptateurs]]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verifier", action="store_true", help="contrôles seulement, sans lancer vLLM")
    args = parser.parse_args()

    config = charger_config()
    verifier_environnement(config)
    chemins = telecharger_adaptateurs(config)
    cmd = commande(config, chemins)
    print("Commande :", shlex.join(cmd), flush=True)               # aucun secret dedans
    if args.verifier:
        return
    os.execvp(cmd[0], cmd)          # vLLM remplace ce processus : il reçoit directement les signaux


if __name__ == "__main__":
    sys.exit(main())
