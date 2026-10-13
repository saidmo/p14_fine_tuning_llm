# POC — Agent IA de triage médical (CHSA)

Fine-tuning de **Qwen3-1.7B-Base** par **SFT avec LoRA**, puis alignement par préférences
(**DPO**), sur un corpus médical bilingue français / anglais ; déploiement d'un endpoint de
démonstration servi par **vLLM** derrière une API **FastAPI**, avec un pipeline **CI/CD GitHub Actions**.

> **Projet pédagogique** (OpenClassrooms, parcours AI Engineer) pour un client fictif, le
> Centre Hospitalier Saint-Aurélien (CHSA). Le modèle n'est **pas** un dispositif médical et ne
> doit pas être utilisé pour une décision clinique réelle. À ce stade, le corpus ne contient
> aucun exemple annoté avec un niveau de priorité : **le triage n'est ni appris ni évalué**, seuls
> les connaissances médicales et le comportement de réponse le sont.

## Avancement

| Semaine | Étape | État |
|---|---|---|
| 1 | Préparation et structuration des données | ✅ dataset publié |
| 2 | Fine-tuning supervisé (SFT) avec LoRA | ✅ adaptateur publié, évalué, model card |
| 3 | Alignement par préférences (DPO) | en cours |
| 4 | Déploiement vLLM / FastAPI, CI/CD, rapport | à venir |

## Dataset (livrable 1)

Publié sur le Hugging Face Hub (dépôt privé) : [`saidmo2/chsa-triage-medical`](https://huggingface.co/datasets/saidmo2/chsa-triage-medical).
La carte complète est versionnée dans [`reports/dataset_card.md`](reports/dataset_card.md).

| Configuration | train | validation | test | eval_clinique |
|---|---|---|---|---|
| `sft` | 5 000 | 500 | 8 821 | 263 |
| `dpo` | 3 000 | 300 | 777 | — |

Sources : MediQAl, FrenchMedMCQA, MedQuAD (SFT) et UltraMedical-Preference (DPO), à des
révisions épinglées dans [`configs/sources.yaml`](configs/sources.yaml).

## Adaptateur SFT (livrable 2)

Publié sur le Hugging Face Hub (dépôt privé) : [`saidmo2/chsa-qwen3-1.7b-sft-lora`](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft-lora).
La model card est versionnée dans [`reports/model_card_sft.md`](reports/model_card_sft.md).

- **LoRA** r 16, alpha 32, sur toutes les couches linéaires (≈ 1 % des paramètres), 2 époques,
  meilleur checkpoint retenu sur la loss de validation (pas 550, eval_loss 0,798).
- **Tokens du chat template** (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`) : jamais
  entraînés dans le modèle Base, leurs vecteurs sont rendus entraînables (PEFT
  `trainable_token_indices`), sans quoi le modèle ne sait pas s'arrêter.
- **Prompt identique à l'inférence** : chat template de Qwen avec `enable_thinking=False` ; la loss
  ne porte que sur la réponse, terminée par `<|im_end|>`.
- **Traçabilité** : révisions du dataset et du modèle, commit Git, configuration et environnement
  enregistrés dans MLflow ; identité bit à bit vérifiée entre le meilleur checkpoint et
  l'adaptateur publié, et conservation des vecteurs appris après fusion.

### Résultats sur le jeu de test (base vs SFT)

Chaque score est comparé à une **référence naïve**, une stratégie sans aucune connaissance
médicale.

| Indicateur | Base | SFT | Référence naïve |
|---|---|---|---|
| Format QCM respecté | 29,6 % | **100 %** | — |
| QCU, exactitude | 1,6 % | 21,7 % | 23,7 % (lettre la plus fréquente) |
| QCM multiples, score partiel | 0,15 | 0,50 | 0,54 (tout cocher) |
| FrenchMedMCQA, exactitude | 0,9 % | **57,9 %** | 26,5 % (lettre la plus fréquente) |
| MedQuAD, ROUGE-L | 0,075 | **0,302** | — |
| Cas cliniques urgences / réanimation, ROUGE-L | 0,072 | 0,075 | — |
| Réponses tronquées (cas cliniques) | 55,5 % | 17,1 % | — |

Le SFT apprend le **comportement attendu** (format, arrêt, style) mais n'apporte pas de gain de
connaissances mesurable sur les QCM cliniques de MediQAl. Le ROUGE-L mesure la forme, pas la
justesse clinique. Détails, ablation de la pénalité de répétition et limites : voir la
[model card](reports/model_card_sft.md).

## Structure

```
configs/          paramètres de chaque étape (données, entraînement, évaluation)
  prive/          paramètres sensibles — NON versionné (voir « Anonymisation »)
data/             raw/ → interim/ → processed/ — NON versionné
reports/          rapports chiffrés (agrégats uniquement), carte du dataset, model card
scripts/          préparation du pod GPU (setup_pod.sh)
src/data/         pipeline de préparation des données
src/training/     entraînement (SFT, puis DPO)
src/evaluation/   évaluation et vérifications de l'adaptateur
tests/            tests (CI)
docker/           images de déploiement
mlflow/, mlruns/  suivi des expériences, rapatrié du pod — NON versionné
```

## Pipeline de données

Chaque étape lit sa configuration dans `configs/`, écrit ses sorties dans `data/` et son
rapport dans `reports/`.

| Étape | Commande | Rôle |
|---|---|---|
| Inventaire | `python -m src.data.explore_sources` | schémas, volumes, licences, révisions |
| Profilage | `python -m src.data.profile_sources` | longueurs en tokens, doublons, fuites, conflits de labels |
| Normalisation | `python -m src.data.normalize` | schéma commun, nettoyage tracé, rejets motivés |
| Sélection | `python -m src.data.select` | anti-fuite test > validation > train, filtres, quotas, tirage stable |
| Mesure ChatDoctor | `python -m src.data.reduce_chatdoctor` | relecture de la réduction des messages de patients réels |
| Anonymisation | `python -m src.data.anonymize` | masquage, réduction ChatDoctor, empreinte des clés |
| Audit exhaustif | `python -m src.data.audit_names` | contrôle de 100 % des textes, décisions humaines |
| Formatage | `python -m src.data.build_dataset` | format conversationnel TRL, contrôle des longueurs |
| Publication | `python -m src.data.publish_dataset` | Hugging Face Hub + carte du dataset |

Chaque étape critique est **fermée par défaut** : la sélection s'arrête si un quota n'est pas
atteint, l'anonymisation si un lieu n'a pas reçu de décision, le formatage et la publication si
l'audit n'est pas au statut « valide ».

## Entraînement et évaluation

Sur un pod GPU (RunPod, NVIDIA A40 48 Go), environnement activé, depuis la racine du dépôt.
Chaque script dispose d'un mode `--smoke` qui valide la chaîne complète en quelques minutes.

| Étape | Commande | Rôle |
|---|---|---|
| SFT | `python -m src.training.sft` | LoRA + tokens du chat template, contrôles, publication de l'adaptateur |
| Évaluation | `python -m src.evaluation.evaluate --modele base` puis `--modele sft` | même protocole pour chaque modèle (QCM, questions ouvertes, latence) |
| Ablation | `python -m src.evaluation.evaluate --modele sft --repetition-penalty 1.0` | effet de la pénalité de répétition |
| Vérification du checkpoint | `python -m src.evaluation.verifier_checkpoint_sft --output-dir <run> --adapter saidmo2/chsa-qwen3-1.7b-sft-lora` | adaptateur publié identique au meilleur checkpoint |
| Vérification de la fusion | `python -m src.evaluation.verifier_adaptateur_sft --adapter saidmo2/chsa-qwen3-1.7b-sft-lora` | vecteurs appris conservés après `merge_and_unload` |

- **Protocole d'évaluation** : génération déterministe (greedy), arrêt sur `<|im_end|>`, pénalité
  de répétition 1.1 sur les réponses libres (valeur fixée a priori, jamais réglée sur le test),
  références naïves pour les QCM, toutes les prédictions conservées pour relecture humaine.
- **MLflow** : expériences `chsa-sft` et `chsa-eval` ; la base et les artefacts sont sur le volume
  persistant du pod, puis rapatriés localement.
- **Rien d'important n'existe seulement sur le pod** : code sur GitHub, adaptateurs sur le Hub,
  suivi MLflow rapatrié.

## Anonymisation (RGPD)

- **Des règles précises masquent** (civilités, lexiques de prénoms, motifs de contexte,
  coordonnées) ; **la NER spaCy sert uniquement à l'audit** : sa précision mesurée sur ces textes
  médicaux (40 % en français, 6 % en anglais) la rendait dangereuse pour l'information clinique.
- **Personnes réelles** (messages ChatDoctor) : réduction structurelle (salutations,
  présentations, provenances, signatures), dates généralisées au mois, lieux masqués sur décision
  humaine, puis contrôle exhaustif avec une décision humaine pour chaque candidat.
- **Paramètres sensibles hors dépôt** : la liste noire et les termes d'exclusion contiennent des
  noms de personnes réelles. Ils sont conservés dans `configs/prive/donnees_sensibles.yaml`,
  ignoré par Git et détenu par le responsable du traitement. Le dépôt ne contient que les
  identifiants (publics) des exemples exclus. Sans ce fichier, le pipeline fonctionne mais ne
  reproduit pas exactement le masquage et l'audit.

Méthode et chiffres détaillés : section « Anonymisation » de la [carte du dataset](reports/dataset_card.md).

## Installation

Chaque environnement n'installe que ce dont il a besoin.

**Préparation des données (Windows)**

```
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_md
hf auth login
```

**Entraînement et évaluation (pod GPU)** : image Runpod PyTorch 2.8.0, volume persistant monté
sur `/workspace`, jeton Hugging Face fourni comme secret (`HF_TOKEN`).

```
cd /workspace/p14_fine_tuning_llm
bash scripts/setup_pod.sh            # à relancer après chaque redémarrage du pod
source /workspace/venv/bin/activate
```

Le disque conteneur étant effacé à chaque arrêt, `scripts/setup_pod.sh` est **idempotent** :
il réinstalle tmux, crée si besoin un environnement virtuel **sur le volume** qui réutilise le
PyTorch de l'image (`--system-site-packages`), installe les dépendances d'entraînement **sans
jamais remplacer ce PyTorch** (contrainte de version), puis vérifie le GPU, le support bf16, les
versions des bibliothèques et la présence du jeton Hugging Face.

Les dépendances sont décrites dans `requirements-train.txt`, avec les versions exactes dans
`requirements-train.lock`. Celles du déploiement (semaine 4) seront décrites dans un fichier
séparé.

## Licences

- **Dataset dérivé** : CC BY 4.0 (la plus restrictive des sources), avec citation de chaque source.
- **Adaptateur SFT** : Apache-2.0 (licence du modèle de base Qwen3), données d'entraînement citées
  dans la model card.
- **Code** : à préciser.
