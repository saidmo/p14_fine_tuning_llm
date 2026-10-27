# POC — Agent IA de triage médical (CHSA)

Fine-tuning de **Qwen3-1.7B-Base** par **SFT avec LoRA**, puis alignement par préférences
(**DPO**), sur un corpus médical bilingue français / anglais ; déploiement d'un endpoint de
démonstration servi par **vLLM** derrière une API **FastAPI**, avec un pipeline **CI/CD GitHub Actions**.

> **Projet pédagogique** (OpenClassrooms, parcours AI Engineer) pour un client fictif, le
> Centre Hospitalier Saint-Aurélien (CHSA). Le modèle n'est **pas** un dispositif médical et ne
> doit pas être utilisé pour une décision clinique réelle. À ce stade, le corpus ne contient
> aucun exemple annoté avec un niveau de priorité : **le triage n'est ni appris ni évalué**, seuls
> les connaissances médicales et le comportement de réponse le sont. La relecture de sécurité
> confirme cette limite : sur 20 cas cliniques d'urgence, **aucune réponse juste** et 30 à 40 %
> de réponses jugées dangereuses, pour le modèle SFT comme pour le modèle DPO.

## Avancement

| Semaine | Étape | État |
|---|---|---|
| 1 | Préparation et structuration des données | ✅ dataset publié |
| 2 | Fine-tuning supervisé (SFT) avec LoRA | ✅ adaptateur publié, évalué, model card |
| 3 | Alignement par préférences (DPO) | ✅ adaptateur publié, évalué, model card ; **servi comme second modèle** (voir « Modèles servis ») |
| 4 | Déploiement vLLM / FastAPI, CI/CD, validation, rapport | ✅ endpoint déployé sur RunPod, validé ; [rapport technique](reports/rapport_technique.pdf) |

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
- **Modèle SFT fusionné** (base + adaptateur, contrôles bloquants, `provenance.json`) :
  [`saidmo2/chsa-qwen3-1.7b-sft`](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft), révision
  `923746e7`. C'est le point de départ et la référence du DPO, et le **modèle servi par défaut**.
  Sa model card, qui documente aussi le DPO, est versionnée dans
  [`reports/model_card.md`](reports/model_card.md).

## Adaptateur DPO (semaine 3)

Publié sur le Hugging Face Hub (dépôt privé) : [`saidmo2/chsa-qwen3-1.7b-sft-dpo-lora`](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft-dpo-lora),
révision `8818097d`, appliqué au modèle SFT fusionné. Sa carte courte est versionnée dans
[`reports/model_card_dpo.md`](reports/model_card_dpo.md).

- **DPO standard** (loss sigmoid, β 0,1), lr 5e-6, 1 époque (188 pas), batch effectif 16 ; LoRA
  identique au SFT ; **référence** = modèle SFT fusionné, adaptateur désactivé ; **tokens du chat
  template gelés** (déjà appris par le SFT).
- **Meilleur checkpoint** sur la loss DPO de validation : pas 175 (0,494) ; les log-probabilités des
  réponses préférées augmentent plus vite que celles des réponses rejetées.
- **Précision des récompenses implicites sur le test DPO : 75,4 %** (n = 777), dont 73,4 % sur
  8 sources jamais vues et 73,6 % sur les préférences annotées par des humains. Calcul indépendant
  de TRL, vérifié contre TRL sur la validation.
- **Non fusionné** : la mise à jour DPO est si petite que la fusion en bf16 efface 94 % de ses
  coefficients (17 % pour le SFT) ; l'adaptateur reste séparé et sera servi tel quel par vLLM.
  Diagnostic : [`reports/verification_fusion_*.json`](reports).

## Modèles servis : SFT par défaut, DPO second modèle

Seuils d'acceptation du DPO **fixés avant son évaluation**, appliqués au modèle tel qu'il sera servi :

| Seuil | SFT | DPO | Résultat |
|---|---|---|---|
| Format QCM ≥ 99 % | 100 % | 100 % | ✅ |
| FrenchMedMCQA, baisse ≤ 3 points | 57,9 % | 56,1 % | ✅ |
| Réponses tronquées ≤ SFT (cas cliniques) | 17,1 % | 20,2 % | ❌ (McNemar p = 0,32, non significatif) |
| Réponses tronquées ≤ SFT (MedQuAD) | 6,7 % | 6,3 % | ✅ |
| Précision des récompenses (test DPO) > 0,5 | — | 75,4 % | ✅ |
| Réponses « dangereuses » ≤ SFT (relecture, 20 cas) | 6 | 8 | ❌ (test du signe p = 0,73, non significatif) |

Le DPO apprend les préférences visées, sans dégrader les QCM ni le français, mais il n'a pas
démontré qu'il était au moins aussi sûr que le SFT. **Décision : le SFT est servi par défaut ;
le DPO est déployé comme second modèle**, accessible explicitement, à des fins de démonstration.

Détails : [`reports/model_card.md`](reports/model_card.md),
[`reports/comparaison_sft_dpo_separe.json`](reports/comparaison_sft_dpo_separe.json),
[`reports/relecture_securite_dpo_separe.json`](reports/relecture_securite_dpo_separe.json).

### Résultats sur le jeu de test (base, SFT, DPO servi)

Chaque score est comparé à une **référence naïve**, une stratégie sans aucune connaissance
médicale.

| Indicateur | Base | SFT | DPO | Référence naïve |
|---|---|---|---|---|
| Format QCM respecté | 29,6 % | **100 %** | **100 %** | — |
| QCU, exactitude | 1,6 % | 21,7 % | 21,3 % | 23,7 % (lettre la plus fréquente) |
| QCM multiples, score partiel | 0,15 | 0,50 | 0,50 | 0,54 (tout cocher) |
| FrenchMedMCQA, exactitude | 0,9 % | **57,9 %** | 56,1 % | 26,5 % (lettre la plus fréquente) |
| MedQuAD, ROUGE-L | 0,075 | **0,302** | 0,302 | — |
| Cas cliniques urgences / réanimation, ROUGE-L | 0,072 | 0,075 | 0,073 | — |
| Réponses tronquées (cas cliniques) | 55,5 % | 17,1 % | 20,2 % | — |

Le SFT apprend le **comportement attendu** (format, arrêt, style) mais n'apporte pas de gain de
connaissances mesurable sur les QCM cliniques de MediQAl ; le DPO n'en apporte pas davantage, ce
qui était attendu (91 % des paires MedQA portent sur l'explication, pas sur la réponse). Le ROUGE-L
mesure la forme, pas la justesse clinique. Détails, ablation de la pénalité de répétition et
limites : voir la [model card](reports/model_card.md).

## Déploiement (livrable 3)

Endpoint de démonstration sur **RunPod**, en deux services :

```
utilisateur ──HTTPS──▶ passerelle FastAPI (pod CPU) ──HTTPS──▶ filtre ──▶ vLLM (pod GPU A40)
   clé X-API-Key        authentification, contrôles,          liste blanche    SFT (défaut)
                        journal SQLite (network volume)       + clé            + adaptateur DPO
```

| Composant | Image (épinglée) | Rôle |
|---|---|---|
| Passerelle | `ghcr.io/saidmo/chsa-passerelle:v0.1.0@sha256:9247ef5d…` (publiée par le CD) | seul point d'entrée des utilisateurs : clé, validation, réponse + priorité, avertissement, journal |
| vLLM | `vllm/vllm-openai:v0.31.0@sha256:a4a4c043…` (CUDA 13.0) | sert `chsa-sft` (défaut) et `chsa-dpo` (LoRA appliqué à l'exécution) |
| Filtre | même pod que vLLM (`src/serving/filtre_vllm.py`) | seules 5 routes autorisées, clé vérifiée ; vLLM n'écoute que sur 127.0.0.1 |

- **API** : `POST /v1/reponses` (question, modèle, priorité), `GET /sante`. Chaque réponse renvoie
  le modèle, la révision des poids, l'identifiant d'interaction, `complete` (réponse tronquée
  signalée) et l'avertissement « hors usage clinique ».
- **Niveau de priorité** : demandé par le descriptif, il est produit par le modèle (QCU sur
  l'échelle FRENCH, 3A et 3B regroupés) mais toujours marqué **`non_validee`** : le modèle n'a
  jamais appris le triage.
- **Configuration unique** : [`configs/serving.yaml`](configs/serving.yaml) (révisions, paramètres
  de génération identiques à l'évaluation, seuils de validation fixés a priori).
- **Sécurité** : quatre clés cloisonnées (RunPod, vLLM, utilisateurs, Hugging Face en lecture
  seule) ; empreintes SHA-256 des clés utilisateurs seulement ; journal sans IP ni clé, purgé à
  30 jours ; images sans secret ni poids.

### Validation du déploiement

Rapport : [`reports/validation_deploiement_20261027-150734.json`](reports/validation_deploiement_20261027-150734.json),
diagnostic : [`reports/diagnostic_accord_20261027-152849.json`](reports/diagnostic_accord_20261027-152849.json).

| Volet | Résultat |
|---|---|
| Prompt servi = prompt d'entraînement | identique au token près |
| Débit (1 requête, vs transformers) | SFT 114 tok/s (×2,8), DPO 98 tok/s (×4,2) |
| Charge (16 requêtes simultanées) | 1 052 tok/s, p95 5,0 s, aucune erreur |
| Exactitude vLLM vs transformers | écart 0,2 et 0,6 point ✅ |
| Accord des lettres QCM (seuil 95 %) | 94,0 % et 93,4 % ❌ — égalités de logits bf16, vLLM ne se reproduit lui-même qu'à 94 % |
| Priorité : format valide | 100 % (mais seulement Tri 1 et Tri 3 : biais de lettre) |
| Robustesse, routes fermées, traçabilité | ✅ (917 / 917 réponses retrouvées dans le journal) |

**Décision (checklist go / no-go du rapport)** : *go* pour la démonstration technique, avec
réserve documentée ; *no-go* pour tout usage clinique.

### Démarrer l'endpoint

1. Pod vLLM (template RunPod `chsa-vllm-premier-lancement`, A40, pilote CUDA ≥ 13.0) :
   `apt-get install -y git`, cloner le dépôt dans `/workspace`, puis
   `nohup python3 -m src.serving.demarrer_vllm > /workspace/vllm.log 2>&1 &`
   (contrôles, téléchargement aux révisions épinglées, vLLM puis filtre ; environ 2 minutes).
2. Pod passerelle (template `chsa-passerelle`, CPU, network volume `chsa-journal` sur `/data`) :
   démarre seul ; `GET /sante` doit répondre `"statut":"ok"`.
3. Arrêter les deux pods hors démonstration.

### CI/CD (GitHub Actions)

- **CI** ([`ci.yml`](.github/workflows/ci.yml), à chaque push) : tests de la passerelle avec un
  faux vLLM, tests de cohérence avec le code d'entraînement, tests du filtre et du démarrage, puis
  construction de l'image et essai de fumée du conteneur. Sans GPU ni secret.
- **CD** ([`cd.yml`](.github/workflows/cd.yml), sur étiquette `vX.Y.Z`) : CI complète, puis
  publication de l'image sur GHCR sous sa version (jamais `latest`, jamais écrasée).

## Structure

```
configs/          paramètres de chaque étape (données, entraînement, évaluation)
  prive/          paramètres sensibles — NON versionné (voir « Anonymisation »)
data/             raw/ → interim/ → processed/, relecture/ — NON versionné
reports/          rapports chiffrés (agrégats uniquement), carte du dataset, model cards
scripts/          préparation du pod GPU (setup_pod.sh)
src/data/         pipeline de préparation des données
src/training/     entraînement (SFT, DPO) et fusion vérifiée
src/evaluation/   évaluation, comparaisons, relecture de sécurité, vérifications
src/api/          passerelle FastAPI (configuration, schémas, client vLLM, journal)
src/serving/      pod vLLM (démarrage, filtre), validation du déploiement, diagnostic
tests/            tests (CI)
docker/           image de la passerelle
.github/workflows CI et CD
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
Les scripts d'entraînement et d'évaluation disposent d'un mode `--smoke` qui valide la chaîne
complète en quelques minutes.

| Étape | Commande | Rôle |
|---|---|---|
| SFT | `python -m src.training.sft` | LoRA + tokens du chat template, contrôles, publication de l'adaptateur |
| Fusion | `python -m src.training.fusionner --base … --adaptateur … --repo …` | fusion, 5 contrôles bloquants, publication privée avec `provenance.json` |
| DPO | `python -m src.training.dpo` | LoRA sur le SFT fusionné, contrôle de génération avant / après, récompenses sur le test, publication |
| Évaluation | `python -m src.evaluation.evaluate --modele {base,sft,dpo,dpo_separe}` | même protocole pour chaque modèle (QCM, questions ouvertes, latence) |
| Ablation | `python -m src.evaluation.evaluate --modele sft --repetition-penalty 1.0` | effet de la pénalité de répétition |
| Comparaison appariée | `python -m src.evaluation.comparer_sft_dpo --sft … --dpo …` | troncatures (McNemar), longueur, langue ; grille de relecture en aveugle |
| Relecture | `python -m src.evaluation.agreger_relecture` puis `relecture_complement` | bilan de la relecture de sécurité, report sur le modèle servi |
| Vérification du checkpoint | `python -m src.evaluation.verifier_checkpoint_sft --output-dir <run> --adapter saidmo2/chsa-qwen3-1.7b-sft-lora` | adaptateur publié identique au meilleur checkpoint |
| Vérification de la fusion | `python -m src.evaluation.verifier_adaptateur_sft --adapter saidmo2/chsa-qwen3-1.7b-sft-lora` | vecteurs appris conservés après `merge_and_unload` |
| Fidélité de la fusion bf16 | `python -m src.evaluation.verifier_fusion --base … --adaptateur … [--preferences]` | part de la mise à jour effacée par l'arrondi, préférences conservées |

- **Protocole d'évaluation** : génération déterministe (greedy), arrêt sur `<|im_end|>` et
  `<|endoftext|>`, pénalité de répétition 1.1 sur les réponses libres (valeur fixée a priori,
  jamais réglée sur le test), références naïves pour les QCM, toutes les prédictions conservées
  pour relecture humaine.
- **Seuils d'acceptation fixés avant chaque évaluation** ; le test n'est jamais utilisé pour un
  choix (checkpoint, hyperparamètre).
- **Relecture de sécurité** : 20 cas cliniques tirés avec une graine fixée, en aveugle (réponses A / B
  dans un ordre aléatoire, clé ouverte après la relecture), grille juste / partiel / faux / dangereux.
  La grille et la clé sont conservées hors Git (`data/relecture/`) ; seuls les agrégats sont versionnés.
- **MLflow** : expériences `chsa-sft`, `chsa-dpo` et `chsa-eval` ; la base et les artefacts sont
  sur le volume persistant du pod, puis rapatriés localement.
- **Rien d'important n'existe seulement sur le pod** : code sur GitHub, modèles et adaptateurs sur
  le Hub, suivi MLflow et prédictions sauvegardés.
- **Moindre privilège** : le pod n'a qu'un jeton GitHub en lecture ; les rapports produits sur le
  pod sont rapatriés et commités depuis le poste de travail.

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
`requirements-train.lock`.

**Passerelle (tests en local)** : environnement séparé, Python 3.12 comme l'image.

```
py -3.12 -m venv .venv-api
.venv-api\Scripts\activate
python -m pip install -r requirements-api.lock pytest
python -m pytest tests -v
```

## Licences

- **Dataset dérivé** : CC BY 4.0 (la plus restrictive des sources), avec citation de chaque source.
- **Adaptateurs SFT et DPO, modèle SFT fusionné** : Apache-2.0 (licence du modèle de base Qwen3),
  données d'entraînement citées dans la model card.
- **Code** : Apache-2.0 (voir [`LICENSE`](LICENSE)).
