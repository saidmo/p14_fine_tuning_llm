# POC — Agent IA de triage médical (CHSA)

Fine-tuning de **Qwen3-1.7B-Base** par **SFT avec LoRA**, puis alignement par préférences
(**DPO**), sur un corpus médical bilingue français / anglais ; déploiement d'un endpoint de
démonstration servi par **vLLM** derrière une API **FastAPI**, avec un pipeline **CI/CD GitHub Actions**.

> **Projet pédagogique** (OpenClassrooms, parcours AI Engineer) pour un client fictif, le
> Centre Hospitalier Saint-Aurélien (CHSA). Le modèle n'est **pas** un dispositif médical et ne
> doit pas être utilisé pour une décision clinique réelle.

## Avancement

| Semaine | Étape | État |
|---|---|---|
| 1 | Préparation et structuration des données | ✅ dataset publié |
| 2 | Fine-tuning supervisé (SFT) avec LoRA | à venir |
| 3 | Alignement par préférences (DPO) | à venir |
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

## Structure

```
configs/        paramètres de chaque étape (sources, sélection, anonymisation, lexiques,
                lieux, décisions d'audit, formatage, publication)
  prive/        paramètres sensibles — NON versionné (voir « Anonymisation »)
data/           raw/ → interim/ → processed/ — NON versionné
reports/        rapports chiffrés de chaque étape (agrégats uniquement) et carte du dataset
src/data/       pipeline de préparation des données
tests/          tests (CI)
docker/         images de déploiement
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

## Installation (préparation des données, Windows)

```
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_md
hf auth login
```

Les dépendances d'entraînement (semaines 2-3) et de déploiement (semaine 4) seront décrites dans
des fichiers séparés, chaque environnement n'installant que ce dont il a besoin.

## Licences

- **Dataset dérivé** : CC BY 4.0 (la plus restrictive des sources), avec citation de chaque source.
- **Code** : à préciser.
