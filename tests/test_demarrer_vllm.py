"""La commande vllm serve est une fonction pure de configs/serving.yaml : testable dans la CI."""
from pathlib import Path, PurePosixPath

from src.api.config import charger_config
from src.serving.demarrer_vllm import commande

CONFIG = charger_config(Path(__file__).resolve().parents[1] / "configs" / "serving.yaml")
CHEMINS = {a.nom_servi: PurePosixPath("/m") / a.nom_servi / a.revision for a in CONFIG.vllm.adaptateurs}
CMD = commande(CONFIG, CHEMINS)


def valeur(option: str) -> str:
    return CMD[CMD.index(option) + 1]


def test_modele_de_base_epingle():
    assert CMD[:3] == ["vllm", "serve", CONFIG.vllm.modele.repo]
    assert valeur("--revision") == CONFIG.vllm.modele.revision
    assert valeur("--tokenizer-revision") == CONFIG.vllm.modele.revision
    assert valeur("--served-model-name") == CONFIG.passerelle.modele_par_defaut


def test_adaptateur_dpo_servi_separe_a_sa_revision():
    dpo = next(a for a in CONFIG.vllm.adaptateurs if a.nom_servi == "chsa-dpo")
    assert "--enable-lora" in CMD
    assert valeur("--max-lora-rank") == str(dpo.rang)
    assert f"chsa-dpo=/m/chsa-dpo/{dpo.revision}" in CMD    # le dossier porte la révision


def test_parametres_du_depot_ignores():
    assert valeur("--generation-config") == "vllm"


def test_cle_d_api_jamais_en_argument():
    assert "--api-key" not in CMD                              # visible dans la liste des processus

def test_image_vllm_epinglee():
    """Maintenant que la version est choisie, la CI refuse toute image non épinglée."""
    CONFIG.vllm.verifier_image()
    assert CONFIG.vllm.version_vllm == "0.31.0"

def test_vllm_injoignable_depuis_l_exterieur():
    assert valeur("--host") == "127.0.0.1"
    assert valeur("--port") == str(CONFIG.vllm.port_interne)
    assert "--disable-fastapi-docs" in CMD