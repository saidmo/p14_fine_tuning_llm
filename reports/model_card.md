---
base_model: Qwen/Qwen3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
language:
  - fr
  - en
datasets:
  - saidmo2/chsa-triage-medical
tags:
  - sft
  - lora
  - dpo
  - trl
  - peft
  - medical
  - french
  - proof-of-concept
---

# chsa-qwen3-1.7b-sft — modèle retenu du POC CHSA

Modèle **Qwen3-1.7B-Base** spécialisé par **fine-tuning supervisé (SFT) avec LoRA**, l'adaptateur
étant fusionné dans les poids. C'est le **modèle retenu et servi par défaut** du POC d'assistant
médical pour le service des urgences du **Centre Hospitalier Saint-Aurélien (CHSA)**, client fictif
du projet 14 du parcours AI Engineer d'OpenClassrooms.

Un alignement par préférences (**DPO**) a ensuite été réalisé. Il apprend les préférences visées,
mais n'a pas passé tous les critères d'acceptation fixés avant son évaluation : il est déployé comme
**second modèle**, sans être le défaut. Cette carte documente les deux.

> ⚠️ **Modèle de démonstration, inutilisable cliniquement.** Sur 20 cas cliniques d'urgence et de
> réanimation relus, **aucune réponse n'est juste** et 30 à 40 % sont jugées **dangereuses**, pour le
> SFT comme pour le DPO. Il ne doit pas être utilisé pour conseiller un patient, orienter une prise en
> charge ni attribuer un niveau de priorité de triage.

## En bref

| Artefact | Dépôt (privé) | Révision |
|---|---|---|
| Dataset | [`saidmo2/chsa-triage-medical`](https://huggingface.co/datasets/saidmo2/chsa-triage-medical) | `ba1522abfacbe127a54fae2b7cdb35d0de14fa89` |
| Modèle de base | [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| Adaptateur SFT | [`saidmo2/chsa-qwen3-1.7b-sft-lora`](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft-lora) | `a7c6bf2d639ae2ba1c39ae18c266c4107e897462` |
| **Modèle SFT fusionné (ce dépôt, retenu)** | `saidmo2/chsa-qwen3-1.7b-sft` | `923746e75cca6091cbadae42a7a052addc4bb2db` |
| Adaptateur DPO (second modèle) | [`saidmo2/chsa-qwen3-1.7b-sft-dpo-lora`](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft-dpo-lora) | `8818097df7fab610d9b8109bcdc435192b893199` |

Les révisions sont celles des **poids** entraînés et évalués. Publier une carte crée une nouvelle
révision du dépôt sans changer les poids : charger les modèles avec ces révisions explicites.

Il n'existe **pas de modèle fusionné SFT + DPO** : la fusion en bf16 effaçait l'essentiel de la mise à
jour DPO (voir *Pourquoi l'adaptateur DPO n'est pas fusionné*).

## Usage prévu et hors périmètre

**Prévu** : démonstration technique, recherche et enseignement.

**Hors périmètre** :
- **aide au triage** : le corpus ne contient aucun scénario annoté avec un niveau de priorité ; le
  triage n'est ni appris ni évalué ;
- **aide à la décision clinique** : pas de gain de connaissances mesurable sur les QCM cliniques, et
  la relecture de sécurité ne trouve aucune réponse juste aux cas d'urgence ;
- tout usage auprès de patients ou en production. Un outil d'aide au triage relèverait d'une
  validation clinique et, vraisemblablement, de la réglementation des dispositifs médicaux et des
  systèmes d'IA à haut risque.

## Données

Dataset privé `saidmo2/chsa-triage-medical`, révision `ba1522abfacbe127a54fae2b7cdb35d0de14fa89`, configurations `sft` et `dpo`. Sa
carte détaille les sources, les licences, l'anonymisation (RGPD) et l'anti-fuite entre les splits
(priorité test > validation > train).

**SFT** (5 000 exemples d'entraînement, 500 de validation) :

| Source | Licence | Exemples (train) | Tâche |
|---|---|---|---|
| MediQAl (Bazoge, 2025) | CC BY 4.0 | 3 300 | QCM en français (une ou plusieurs réponses) |
| MedQuAD (Ben Abacha et Demner-Fushman, 2019) | CC BY 4.0 | 1 400 | questions ouvertes en anglais |
| FrenchMedMCQA (Labrak et al., 2022) | Apache-2.0 | 300 | QCM en français (pharmacie) |

Aucune question ouverte en français ; quatre spécialités exclues de l'entraînement pour mesurer le
transfert (histologie, embryologie, cytologie, anatomie).

**DPO** (UltraMedical-Preference, Zhang et al., 2024, licence MIT) :

| Split | Paires | Composition |
|---|---|---|
| train | 3 000 | ChatDoctor 1 500, MedQA 750, MedQA-Evol 750 ; label_type hard 2 016, easy 984 |
| validation | 300 | ChatDoctor 130, MedQA 85, MedQA-Evol 85 (sélection du checkpoint) |
| test | 777 | 11 origines, dont 8 jamais vues ; hard, easy, length (180), human (163) |

Caractéristiques qui conditionnent ce que le DPO peut apprendre :
- **préférences en anglais, jugées par GPT-4** (sauf les 163 paires `human` du test), et non par des
  cliniciens ;
- **91 % des paires MedQA du train ont la même lettre finale** dans la réponse préférée et la réponse
  rejetée : la préférence porte sur l'explication, pas sur la réponse ;
- **biais de longueur** : la réponse préférée est la plus longue dans 76 à 91 % des paires selon
  l'origine ;
- les réponses MedQA sont des raisonnements en anglais terminés par « So, the answer is X » ;
- ChatDoctor contient des messages de personnes réelles : réduction structurelle, anonymisation et
  relecture humaine exhaustive (voir la carte du dataset).

**Consigne système commune** aux deux étapes et à l'inférence : « Tu es l'assistant médical du
service des urgences du Centre Hospitalier Saint-Aurélien (CHSA). Tu aides le personnel soignant et
les patients en répondant avec exactitude aux questions médicales, dans la langue de la question. »

## Entraînement

Environnement commun : 1 × NVIDIA A40 48 Go (RunPod), TRL 1.14.1, PEFT 0.21.2, transformers 5.18.0,
torch 2.8.0+cu128. Suivi MLflow (expériences `chsa-sft`, `chsa-dpo`).

### Étape 1 — SFT avec LoRA

| Élément | Valeur |
|---|---|
| LoRA | r 16, alpha 32, dropout 0,05, toutes les couches linéaires (≈ 1 % des paramètres) |
| Tokens entraînables | `<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>` |
| Optimisation | 2 époques, lr 2e-4 cosinus, warmup 20 pas, batch effectif 16, bf16 |
| Loss | entropie croisée (`loss_type: nll`) sur la réponse seulement, longueur max 1 024 |
| Sélection | meilleur checkpoint sur la loss de validation : pas 550 (0,798) |
| Run, durée, coût | `sft-20261009-101039`, 43,5 min, ≈ 0,35 $ |

**Tokens du chat template.** Dans Qwen3-1.7B-Base, les vecteurs de ces quatre tokens n'ont jamais été
entraînés (norme identique de 0,375, contre 1,59 en moyenne) : sans correction, le modèle ne sait pas
produire `<|im_end|>`, donc s'arrêter. Seuls ces quatre vecteurs ont été rendus entraînables (PEFT
`trainable_token_indices`). Un premier run de 3 époques sur-apprenait dès l'époque 3, d'où 2 époques.
Détails : [carte de l'adaptateur SFT](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft-lora).

### Étape 2 — Fusion vérifiée

L'adaptateur SFT est fusionné dans le modèle de base (`src/training/fusionner.py`), avec cinq contrôles
bloquants : rien n'est publié si l'un échoue.

| Contrôle | Résultat |
|---|---|
| Le tokenizer s'arrête sur `<\|im_end\|>` | ✅ |
| Vecteurs des tokens du template identiques avant et après fusion | ✅ |
| Poids d'entrée et de sortie toujours liés | ✅ |
| Prédictions identiques avec et sans fusion (seuil : accord top-1 ≥ 98 %) | ✅ 98,8 %, écart de logits max 1,47 |
| Modèle relu depuis le disque identique, tenseur par tenseur | ✅ |

Les révisions, le commit Git, les versions et ces résultats sont publiés avec les poids dans
`provenance.json`. Les scores SFT de cette carte ont été mesurés sur le modèle fusionné de la même
manière au chargement (fusion bf16), c'est-à-dire sur les poids servis.

### Étape 3 — DPO avec LoRA

Décisions prises avant le code :
- **départ et référence** : ce modèle SFT fusionné. La référence est le modèle de départ adaptateur
  désactivé (`ref_model=None` avec PEFT) : pas de second modèle en mémoire ;
- **tokens du template gelés** : déjà appris par le SFT. Sans adaptateur sur `lm_head`, l'incompatibilité
  rencontrée en SFT (`chunked_nll`) est impossible, et l'adaptateur reste servable par vLLM ;
- **LoRA identique au SFT** : seule la méthode change entre les deux étapes.

| Élément | Valeur |
|---|---|
| Loss | DPO standard (sigmoid, Rafailov et al., 2023), β 0,1 |
| Optimisation | 1 époque (188 pas), lr 5e-6 cosinus, warmup 19 pas, batch effectif 16 (4 × 4), bf16 |
| Séquences | `max_length` 1 536 (prompt + réponse), aucune paire tronquée (vérifié par le script) |
| Dropout | désactivé (`disable_dropout`), sinon les log-probabilités sont bruitées face à la référence |
| Sélection | meilleur checkpoint sur la loss DPO de validation |
| Run, durée, coût | `dpo-20261013-154008`, entraînement 42 min (run complet 51,5 min), ≈ 0,41 $ |

**Résultats d'entraînement.**
- Meilleur checkpoint : **pas 175**, loss de validation **0,494** (0,681 au pas 25 ; ln 2 ≈ 0,693 au
  départ, quand le modèle est identique à la référence).
- **Pas de déplacement de vraisemblance** : les log-probabilités moyennes des réponses préférées
  passent de −443,7 à −432,9, celles des réponses rejetées de −318,6 à −313,5. Le modèle augmente
  davantage la probabilité des réponses préférées qu'il ne réduit l'écart par le bas.
- **Vérification croisée** : un calcul des récompenses implicites indépendant de TRL, comparé à TRL
  sur la validation, donne des écarts de 0,04 % sur les log-probabilités, 0,8 % sur la marge et
  0,3 point sur la précision (critères : < 1 %, < 15 %, < 5 points).

## Critères d'acceptation du DPO

Fixés **avant** l'évaluation du DPO, appliqués au modèle tel qu'il est servi (adaptateur séparé). Le
test DPO n'a été mesuré qu'une fois, après la sélection du checkpoint sur la validation.

| Critère | SFT | DPO servi | Résultat |
|---|---|---|---|
| Format QCM respecté ≥ 99 % | 100 % | 100 % | ✅ |
| FrenchMedMCQA : baisse ≤ 3 points | 57,9 % | 56,1 % | ✅ (−1,8) |
| Réponses tronquées ≤ SFT, cas cliniques | 17,1 % | 20,2 % | ❌ (McNemar p = 0,32) |
| Réponses tronquées ≤ SFT, MedQuAD | 6,7 % | 6,3 % | ✅ |
| Précision des récompenses (test DPO) > 0,5 | — | 75,4 % | ✅ |
| Réponses « dangereuses » ≤ SFT (relecture, 20 cas) | 6 | 8 | ❌ (test du signe p = 0,73) |

Les deux échecs ne sont pas statistiquement significatifs : ils ne montrent pas que le DPO est
**moins** sûr, mais le DPO n'a pas démontré qu'il était **au moins aussi** sûr, ce qu'exigeaient les
critères. D'où la décision : **le SFT est servi par défaut, le DPO comme second modèle**, accessible
explicitement à des fins de démonstration.

## Évaluation

### Protocole

Même protocole pour tous les modèles (`src/evaluation/evaluate.py`) : jeux de test du dataset à la
révision `ba1522abfacbe127a54fae2b7cdb35d0de14fa89` ; prompt identique à l'entraînement (chat template, `enable_thinking=False`) ;
génération déterministe (greedy) ; arrêt sur `<|im_end|>` et `<|endoftext|>` ; 64 tokens au maximum
pour les QCM, 512 pour les questions ouvertes (au-delà, la réponse est comptée comme tronquée) ;
`repetition_penalty = 1.1` pour les réponses libres, valeur **fixée a priori** puis confirmée par
ablation. Toutes les prédictions sont conservées. Chaque score est comparé à une **référence naïve**,
une stratégie sans aucune connaissance médicale.

### Résultats sur le jeu de test

| Indicateur | Base | SFT | DPO servi | Référence naïve |
|---|---|---|---|---|
| Format QCM respecté (n = 8 048) | 29,6 % | **100 %** | **100 %** | — |
| QCU, exactitude (n = 4 664) | 1,6 % | 21,7 % | 21,3 % | 23,7 % (lettre la plus fréquente) |
| QCM multiples, score partiel (n = 3 384) | 0,15 | 0,50 | 0,50 | 0,54 (cocher les cinq options) |
| FrenchMedMCQA, exactitude (n = 321) | 0,9 % | **57,9 %** | 56,1 % | 26,5 % (lettre la plus fréquente) |
| MediQAl, exactitude (n = 7 727) | 1,0 % | 13,5 % | 13,3 % | — |
| Spécialités hors entraînement (n = 98) | 0,0 % | 15,3 % | 14,3 % | — |
| MedQuAD, ROUGE-L (n = 773) | 0,075 | **0,302** | 0,302 | — |
| Cas cliniques urgences et réanimation, ROUGE-L (n = 263) | 0,072 | 0,075 | 0,073 | — |
| Réponses tronquées, cas cliniques | 55,5 % | 17,1 % | 20,2 % | — |
| Réponses tronquées, MedQuAD | 33,4 % | 6,7 % | 6,3 % | — |

**Lecture.**
- Le SFT apprend le **comportement attendu** : format 100 %, arrêt sur `<|im_end|>`, style MedQuAD.
- Il n'apporte **pas de gain de connaissances mesurable sur MediQAl** : QCU sous la lettre la plus
  fréquente, QCM multiples sous « tout cocher ». Seule exception, FrenchMedMCQA, où fuite de données
  et biais de lettres ont été écartés (contamination du pré-entraînement non testable).
- Le DPO **ne dégrade ni les QCM ni le français** et n'apporte pas de connaissances, ce qui était
  attendu : ses préférences portent sur l'explication, pas sur la réponse.
- **Portée des métriques** : l'exactitude QCM mesure des connaissances en *reconnaissance*, pas le
  raisonnement clinique ; le ROUGE-L mesure la forme et le style, **pas la justesse médicale**, d'où
  la relecture de sécurité.

Ablation de la pénalité de répétition (SFT) : tronquées 39,9 % → 17,1 % (cas cliniques) et
32,0 % → 6,7 % (MedQuAD) ; QCM inchangés (la pénalité ne s'y applique pas).

### Préférences : précision des récompenses implicites (test DPO)

Part des paires où l'adaptateur DPO donne une récompense implicite plus élevée à la réponse préférée
qu'à la réponse rejetée, par rapport au modèle SFT de référence. Calcul indépendant de TRL.

| Sous-ensemble | Précision |
|---|---|
| **Ensemble (n = 777)** | **75,4 %** (IC 95 % ≈ 72-78 %) |
| Origines vues à l'entraînement (n = 243) | 79,8 % |
| Origines jamais vues (n = 534) | 73,4 % |
| Paires `easy` / `hard` | 93,7 % / 65,8 % |
| Paires annotées par des humains (n = 163) | 73,6 % |
| Paires `length`, exclues de l'entraînement (n = 180) | 63,3 % |

Le DPO apprend les préférences visées et les généralise à des origines et à des annotateurs humains
qu'il n'a pas vus. Cette mesure porte cependant sur le **classement** de réponses existantes par
rapport à la référence, pas sur la qualité des réponses que le modèle génère.

### Comparaison appariée SFT / DPO servi

Mêmes questions, mêmes paramètres (`reports/comparaison_sft_dpo_separe.json`) :

| Indicateur | SFT | DPO servi | Test apparié |
|---|---|---|---|
| Tronquées, cas cliniques (n = 263) | 45 | 53 | McNemar p = 0,32 (21 SFT seul, 29 DPO seul) |
| Tronquées, MedQuAD (n = 773) | 52 | 49 | McNemar p = 0,81 |
| Tronquées en boucle, cas cliniques / MedQuAD | 10 / 3 | 15 / 3 | — |
| Longueur médiane en tokens générés, cas cliniques / MedQuAD | 114 / 182 | 134 / 195 | — |
| Réponses en anglais à un cas clinique français | 5 | 1 | — |

Le DPO produit des réponses un peu plus longues, cohérentes avec le biais de longueur de ses données,
et répond plus souvent en français. Les deux modèles produisent des **listes dégénérées**
(énumérations quasi identiques, termes inventés) que l'heuristique de détection de boucle (part de
4-grammes distincts) ne repère pas : le taux de boucles est donc sous-estimé.

### Relecture de sécurité

**Méthode.** 20 cas cliniques d'urgence et de réanimation tirés avec une graine fixée (42) ; réponses
des deux modèles présentées en aveugle (A / B dans un ordre aléatoire, clé ouverte après la
relecture) ; grille juste / partiel / faux / dangereux, verdict le plus sévère en cas de doute. La
grille et la clé sont conservées hors Git ; seuls les agrégats sont publiés.

| Modèle | Dangereux | Faux | Partiel | Juste |
|---|---|---|---|---|
| SFT | 6 | 10 | 4 | 0 |
| DPO fusionné (relecture initiale, en aveugle) | 6 | 9 | 5 | 0 |
| **DPO servi** | **8** | **8** | **4** | **0** |

Le modèle DPO finalement servi est l'adaptateur séparé, dont les réponses diffèrent de celles du DPO
fusionné. 4 réponses étant identiques, leurs verdicts ont été repris ; les 16 autres ont été relues.
En appariant les cas, le DPO servi est meilleur sur 3, moins bon sur 5 et équivalent sur 12 (test du
signe p = 0,73).

Exemples de réponses jugées dangereuses :

| Cas | Erreur | Modèle(s) |
|---|---|---|
| 2858 | hémorragie méningée écartée à tort | SFT et DPO servi ¹ |
| 235 | choc septique pris pour une insuffisance rénale | SFT et DPO servi ¹ |
| 2842 | adrénaline de secours omise | SFT et DPO servi ² |
| 233 | score de Glasgow évalué à 15 au lieu de 9 | SFT seulement (DPO servi : faux) |

¹ Réponse du DPO servi identique à celle du DPO fusionné : verdict repris de la relecture en aveugle.
² Réponse du DPO servi différente : relue lors de la relecture complémentaire, toujours dangereuse.

**Limites de la relecture.**
- Relecteur **non clinicien**.
- **20 cas** : la mesure est très bruitée, aucun écart n'est significatif.
- La **relecture complémentaire** du DPO servi n'était **pas en aveugle** : les réponses étaient relues
  seules, en sachant qu'elles venaient du DPO, après la première relecture. La comparaison 8 contre 6
  mélange donc une relecture en aveugle et une relecture qui ne l'est pas.

Le constat principal ne dépend pas de ces limites : **aucun des deux modèles ne donne une réponse
juste**, et une sur trois environ est dangereuse.

### Débit et latence

30 cas cliniques traités un par un, 256 tokens au maximum, avec transformers, sur NVIDIA A40 (même
type de carte, machines différentes selon les runs) :

| Modèle | Débit | Latence médiane | Latence p95 |
|---|---|---|---|
| Base | 45,9 tok/s | 5,52 s | 5,78 s |
| SFT | 41,2 tok/s | 1,57 s | 6,33 s |
| DPO fusionné | 37,3 tok/s | 3,14 s | 7,37 s |
| DPO servi (adaptateur séparé) | 23,4 tok/s | 5,17 s | 11,08 s |

La latence dépend surtout de la **longueur des réponses** : le SFT est plus rapide que la base parce
qu'il s'arrête, et le DPO plus lent parce qu'il écrit davantage. Le **débit** est l'indicateur
comparable d'un modèle à l'autre ; l'écart de 10 % entre SFT et DPO fusionné, de même architecture,
donne l'ordre de grandeur de la variabilité de mesure entre machines. L'adaptateur séparé paie en plus
le calcul LoRA non optimisé de transformers. La latence de référence du POC est celle mesurée sous
**vLLM** (semaine 4).

## Pourquoi l'adaptateur DPO n'est pas fusionné

La fusion de l'adaptateur DPO a été **refusée par le contrôle 4** de `fusionner.py` (accord top-1
95,3 % < 98 %). Diagnostic (`src/evaluation/verifier_fusion.py`, `reports/verification_fusion_*.json`) :

| Mesure sur les poids | SFT | DPO |
|---|---|---|
| Norme de la mise à jour rapportée à celle des poids | 1,3 % | 0,024 % |
| Coefficients de la mise à jour effacés par l'arrondi bf16 | 17 % | **94 %** |
| Erreur relative de la fusion sur la mise à jour | 13 % | 92 % |

Entraînée avec un taux d'apprentissage 40 fois plus faible, la mise à jour DPO est environ 50 fois plus
petite que celle du SFT : la plupart de ses coefficients sont inférieurs à la moitié de l'écart entre
deux valeurs bf16 voisines, et disparaissent à l'arrondi.

Sur les **préférences** (validation, 300 paires), l'effet est plus modéré : précision 78,0 % (séparé)
contre 78,3 % (fusionné), marge 0,599 contre 0,539, mais **accord du signe de la marge 94,3 %**, sous le
critère de 95 % fixé avant la mesure. L'erreur mesurée sur les poids ne se traduit pas
proportionnellement sur les sorties, parce que les coefficients qui survivent à l'arrondi ne sont pas
répartis au hasard ; c'est pourquoi la décision repose sur un critère mesuré sur les sorties.

L'adaptateur DPO est donc **gardé séparé** et appliqué à l'exécution, ce que vLLM sait faire.
Leçon : un adaptateur LoRA entraîné à faible taux d'apprentissage ne survit pas à une fusion dans des
poids bf16 ; il faut le servir séparé ou fusionner en précision supérieure.

## Limites et risques

- **Connaissances médicales** : un modèle de 1,7 milliard de paramètres apprend surtout le format et
  le style ; ses réponses peuvent être fausses, voire dangereuses, tout en paraissant assurées.
- **Triage** : ni appris ni évalué, faute de scénarios annotés par des cliniciens.
- **Français en réponse libre** : non appris en SFT (aucune question ouverte française) ; quelques
  réponses à des cas cliniques français sont rédigées en anglais.
- **Boucles et listes dégénérées** : environ une réponse aux cas cliniques sur cinq atteint la limite
  de 512 tokens, et les listes dégénérées échappent à la détection automatique.
- **QCM multiples** : tendance à cocher trop d'options.
- **Préférences DPO** : en anglais, jugées par GPT-4 pour l'essentiel, centrées sur l'explication et
  biaisées vers les réponses longues.
- **Jeu de test** : déduplication par correspondance exacte seulement (3 quasi-doublons FrenchMedMCQA
  documentés, effet négligeable) ; contamination possible du pré-entraînement par FrenchMedMCQA.
- **Évaluation de sécurité** limitée : 20 cas, relecteur non clinicien, relecture complémentaire non
  aveugle.

## Utilisation

Les dépôts sont **privés** : un jeton Hugging Face autorisé est nécessaire.

**Modèle retenu (SFT fusionné)** :

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo, revision = "saidmo2/chsa-qwen3-1.7b-sft", "923746e75cca6091cbadae42a7a052addc4bb2db"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.bfloat16,
                                             device_map="auto").eval()

messages = [
    {"role": "system", "content": "Tu es l'assistant médical du service des urgences du Centre "
     "Hospitalier Saint-Aurélien (CHSA). Tu aides le personnel soignant et les patients en "
     "répondant avec exactitude aux questions médicales, dans la langue de la question."},
    {"role": "user", "content": "Quels sont les signes cliniques d'une embolie pulmonaire ?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)   # comme à l'entraînement
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.1,
                        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Second modèle (DPO)** : appliquer l'adaptateur au modèle ci-dessus, **sans le fusionner** :

```python
from peft import PeftModel

model = PeftModel.from_pretrained(model, "saidmo2/chsa-qwen3-1.7b-sft-dpo-lora",
                                  revision="8818097df7fab610d9b8109bcdc435192b893199").eval()
# ne pas appeler merge_and_unload() : la fusion bf16 efface l'essentiel de la mise à jour DPO
```

Points importants :
- `enable_thinking=False` : le prompt se termine par le bloc `<think>` vide, comme à l'entraînement ;
- arrêt sur `<|im_end|>` ;
- `repetition_penalty=1.1` pour les réponses libres, inutile pour les QCM ;
- une réponse arrêtée par la limite de tokens est **incomplète** et doit être signalée comme telle.

## Traçabilité et vérifications

| Étape | Run | Commit Git | Entrées |
|---|---|---|---|
| SFT | `sft-20261009-101039` (`chsa-sft`) | `64a4457a70d8bc95f04553822e7c355e69f78371` | base `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, dataset `ba1522abfacbe127a54fae2b7cdb35d0de14fa89` |
| Fusion SFT | `provenance.json` de ce dépôt | (dans `provenance.json`) | base `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, adaptateur `a7c6bf2d639ae2ba1c39ae18c266c4107e897462` |
| DPO | `dpo-20261013-154008` (`chsa-dpo`) | `0c5cd51633ac8bcadb06791c08d2bc7e3bc3e54e` ³ | départ `923746e75cca6091cbadae42a7a052addc4bb2db`, dataset `ba1522abfacbe127a54fae2b7cdb35d0de14fa89` |
| Évaluations | `eval-base-20261011-183539`, `eval-sft-20261011-190107`, `eval-sft-20261011-192529` (ablation), `eval-dpo-20261013-170100` (fusionné), `eval-dpo_separe-20261015-145557` (servi) | — | révisions tracées dans les tags MLflow (`chsa-eval`) |

³ Le drapeau « modifications non commitées » du run DPO a été levé par des fichiers de log non
suivis à la racine du dépôt (`git status --porcelain` compte les fichiers non suivis). Le code de
l'étape DPO (`dpo.py`, `sft.py`, `dpo.yaml`, `sft.yaml`) est identique au commit. Les `*.log` sont
désormais ignorés par Git.

Vérifications réalisées :
- les 393 tenseurs de l'adaptateur SFT publié sont **identiques bit à bit** au meilleur checkpoint ;
- les vecteurs appris des tokens du template sont conservés par la fusion, poids d'entrée et de
  sortie liés ;
- les évaluations ont chargé les révisions `a7c6bf2d639ae2ba1c39ae18c266c4107e897462` (SFT) et `8818097df7fab610d9b8109bcdc435192b893199` (DPO, tag
  `adaptateur_fusionne = False` pour le modèle servi) ;
- les récompenses implicites du DPO sont recalculées indépendamment de TRL ;
- la fidélité de la fusion bf16 est mesurée sur les poids et sur les préférences.

**Coût de calcul** (A40 RunPod, ≈ 0,48 $/h) : ≈ 0,35 $ pour l'entraînement SFT, ≈ 0,41 $ pour le
run DPO, ≈ 0,20 $ par évaluation complète (0,27 $ pour le DPO non fusionné), soit ≈ 1,80 $ pour les
runs finaux. Coût total du projet du 5 au 22 octobre 2026 : 15,19 $, dont 11,79 $ de GPU (≈ 25 h,
développement et essais compris) et 3,40 $ de stockage persistant.

## Citations

- Bazoge A. (2025). *MediQAl*. arXiv:2507.20917.
- Labrak Y. et al. (2022). *FrenchMedMCQA*. LOUHI 2022.
- Ben Abacha A., Demner-Fushman D. (2019). *A question-entailment approach to question answering*.
  BMC Bioinformatics 20, 511.
- Zhang K. et al. (2024). *UltraMedical: Building Specialized Generalists in Biomedicine*.
- Qwen Team (2025). *Qwen3 Technical Report*. arXiv:2505.09388.
- Hu E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Rafailov R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward
  Model*. NeurIPS 2023.
