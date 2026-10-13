---
base_model: Qwen/Qwen3-1.7B-Base
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
  - fr
  - en
datasets:
  - saidmo2/chsa-triage-medical
tags:
  - lora
  - sft
  - trl
  - peft
  - medical
  - french
  - proof-of-concept
---

# chsa-qwen3-1.7b-sft-lora — adaptateur LoRA SFT (POC CHSA)

Adaptateur **LoRA** de **Qwen3-1.7B-Base**, obtenu par **fine-tuning supervisé (SFT)** sur des
questions médicales en français et en anglais. Il constitue la première étape d'un POC d'assistant
médical pour le service des urgences du **Centre Hospitalier Saint-Aurélien (CHSA)**, client fictif
du projet 14 du parcours AI Engineer d'OpenClassrooms. L'étape suivante est un alignement par DPO.

> ⚠️ **Modèle de démonstration, sans validation clinique.** Il ne doit pas être utilisé pour
> conseiller un patient, orienter une prise en charge ni attribuer un niveau de priorité de triage.

## Usage prévu et hors périmètre

**Prévu** : démonstration technique, recherche et enseignement ; base de départ pour l'étape DPO
du projet.

**Hors périmètre** :
- **aide au triage** : le corpus ne contient aucun scénario annoté avec un niveau de priorité, et
  cette capacité n'a pas été évaluée ;
- **aide à la décision clinique** : l'évaluation ne montre pas de gain de connaissances médicales
  mesurable sur les QCM cliniques (voir *Évaluation*) ;
- tout usage auprès de patients ou en production. Un outil d'aide au triage relèverait d'une
  validation clinique et, vraisemblablement, de la réglementation des dispositifs médicaux et des
  systèmes d'IA à haut risque.

## Données d'entraînement

Dataset privé [`saidmo2/chsa-triage-medical`](https://huggingface.co/datasets/saidmo2/chsa-triage-medical),
configuration `sft`, révision `ba1522abfacbe127a54fae2b7cdb35d0de14fa89`. Sa carte
détaille les sources, les licences, l'anonymisation et l'anti-fuite entre les splits.

| Source | Licence | Exemples (train) | Tâche |
|---|---|---|---|
| MediQAl (Bazoge, 2025) | CC BY 4.0 | 3 300 | QCM en français (une ou plusieurs réponses) |
| MedQuAD (Ben Abacha et Demner-Fushman, 2019) | CC BY 4.0 | 1 400 | questions ouvertes en anglais |
| FrenchMedMCQA (Labrak et al., 2022) | Apache-2.0 | 300 | QCM en français (pharmacie) |

- **5 000 exemples** d'entraînement et 500 de validation. Tâches : 2 430 QCU, 1 170 QCM à réponses
  multiples, 1 400 questions ouvertes. Langues : 3 600 en français, 1 400 en anglais.
- **Aucune question ouverte en français** : le français n'est présent que sous forme de QCM.
- **Spécialités exclues de l'entraînement** pour mesurer le transfert : histologie, embryologie,
  cytologie, anatomie.
- **Format des réponses apprises** : `Réponse : B. <libellé>` pour un QCU ; `Réponses : A, D`
  suivi du détail des options pour un QCM ; texte libre pour les questions ouvertes.
- **Consigne système commune** : « Tu es l'assistant médical du service des urgences du Centre
  Hospitalier Saint-Aurélien (CHSA). Tu aides le personnel soignant et les patients en répondant
  avec exactitude aux questions médicales, dans la langue de la question. »

## Entraînement

| Élément | Valeur |
|---|---|
| Modèle de base | `Qwen/Qwen3-1.7B-Base`, révision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| Méthode | SFT avec TRL 1.14.1 et PEFT 0.21.2 (transformers 5.18.0, torch 2.8.0) |
| LoRA | r 16, alpha 32, dropout 0,05, sur q, k, v, o, gate, up et down (toutes les couches linéaires) |
| Tokens entraînables | `<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>` (voir ci-dessous) |
| Paramètres entraînables | ≈ 17,4 M sur 1,74 Md (1,0 %) |
| Optimisation | 2 époques, lr 2e-4 cosinus, 20 pas de warmup, batch effectif 16 (8 × 2), bf16 |
| Séquences | longueur max 1 024 tokens, sans packing |
| Loss | entropie croisée (`loss_type: nll`), sur la réponse seulement (`completion_only_loss`) |
| Sélection | meilleur checkpoint sur la loss de validation (`load_best_model_at_end`) |
| Matériel et coût | 1 × NVIDIA A40 48 Go (RunPod), 626 pas, 43,5 min, ≈ 0,35 $ |
| Suivi | MLflow, expérience `chsa-sft`, run `sft-20261009-101039` |
| Code | GitHub `saidmo/p14_fine_tuning_llm`, commit `64a4457a70d8bc95f04553822e7c355e69f78371` (sans modification non commitée) |

**Mise en forme du prompt.** Le prompt est construit avec le chat template de Qwen et
`enable_thinking=False`, exactement comme à l'inférence : il se termine par le bloc `<think>` vide.
La loss ne porte que sur la réponse, suivie de `<|im_end|>`.

**Tokens du chat template.** Dans Qwen3-1.7B-Base, les vecteurs des quatre tokens du chat template
n'ont jamais été entraînés : ils ont tous une norme de 0,375, identique, contre 1,59 en moyenne
pour les autres tokens. Le modèle ne pouvait donc ni produire `<|im_end|>`, c'est-à-dire s'arrêter,
ni distinguer `<|im_start|>` de `<|im_end|>`. LoRA ne modifiant pas la matrice d'embedding, seuls
ces quatre vecteurs ont été rendus entraînables (`trainable_token_indices` de PEFT). Les poids
d'entrée et de sortie étant liés, le même vecteur sert à lire et à produire chaque token. Normes
après entraînement : 0,392, 0,436, 0,401 et 0,400.

**Courbe de validation.** Un premier run de 3 époques montrait un sur-apprentissage dès le début de
l'époque 3, d'où le choix de 2 époques. La loss de validation baisse régulièrement de 0,867 (pas 50)
à 0,798 (pas 550), et l'essentiel est appris pendant la première époque.

| Indicateur (meilleur checkpoint, pas 550) | Valeur |
|---|---|
| Loss de validation | 0,7980 |
| Précision par token (validation) | 0,8008 |
| Loss d'entraînement moyenne | 0,8188 |

## Évaluation

**Protocole** : jeu de test du dataset, révision exacte ; même prompt qu'à l'entraînement ;
génération déterministe (greedy) ; arrêt sur `<|im_end|>` ; 64 tokens au maximum pour les QCM, 512
pour les questions ouvertes. Pour les questions ouvertes et la latence, `repetition_penalty = 1.1`,
une valeur **fixée a priori** et non réglée sur le test. Le modèle évalué est le modèle de base
fusionné avec cet adaptateur, à la révision `a7c6bf2d639ae2ba1c39ae18c266c4107e897462`, identique
bit à bit au checkpoint du pas 550. Suivi dans MLflow, expérience `chsa-eval`.

Chaque score est comparé à une **référence naïve**, c'est-à-dire une stratégie sans aucune
connaissance médicale :

| Indicateur | Base | SFT | Référence naïve |
|---|---|---|---|
| Format QCM respecté (n = 8 048) | 29,6 % | **100 %** | — |
| QCU, exactitude (n = 4 664) | 1,6 % | 21,7 % | 23,7 % (lettre la plus fréquente) |
| QCM multiples, score partiel (n = 3 384) | 0,15 | 0,50 | 0,54 (cocher les cinq options) |
| QCM multiples, exactitude | 0,2 % | 6,4 % | — |
| FrenchMedMCQA, exactitude (n = 321, QCU) | 0,9 % | **57,9 %** | 26,5 % (lettre la plus fréquente) |
| MediQAl, exactitude (n = 7 727) | 1,0 % | 13,5 % | — |
| Spécialités hors entraînement (n = 98) | 0,0 % | 15,3 % | — |
| MedQuAD, ROUGE-L (n = 773) | 0,075 | **0,302** | — |
| Cas cliniques urgences et réanimation, ROUGE-L (n = 263) | 0,072 | 0,075 | — |
| Réponses tronquées, cas cliniques | 55,5 % | 17,1 % | — |
| Réponses tronquées, MedQuAD | 33,4 % | 6,7 % | — |
| Latence médiane / débit (A40, transformers) | 5,5 s / 46 tok/s | 1,6 s / 41 tok/s | — |

**Ablation de la pénalité de répétition** (même adaptateur, avec et sans pénalité) :

| Indicateur | Sans (1,0) | Avec (1,1) |
|---|---|---|
| Réponses tronquées, cas cliniques | 39,9 % | 17,1 % |
| Réponses tronquées, MedQuAD | 32,0 % | 6,7 % |
| ROUGE-L, cas cliniques | 0,070 | 0,075 |
| ROUGE-L, MedQuAD | 0,265 | 0,302 |

**Lecture.**
- Le SFT apprend le **comportement attendu** : format de réponse respecté à 100 %, arrêt sur
  `<|im_end|>`, style des réponses MedQuAD.
- Il n'apporte **pas de gain de connaissances mesurable sur MediQAl** : les QCU restent sous la
  référence « lettre la plus fréquente », et les QCM multiples sous la référence « tout cocher »,
  le modèle cochant trop d'options.
- Sur **FrenchMedMCQA**, le gain est réel et dépasse nettement la référence. On a vérifié qu'il ne
  s'explique ni par une fuite de données ni par un biais de lettres. Ces questions de pharmacie sont
  courtes, factuelles et proches de la distribution d'entraînement.
- La **latence** baisse parce que les réponses s'arrêtent ; le débit reste le même.

**Portée des métriques.** L'exactitude QCM mesure des connaissances factuelles en *reconnaissance*,
pas le raisonnement clinique. Le ROUGE-L mesure un recouvrement de mots : il renseigne sur la forme
et le style, pas sur la justesse médicale. La qualité clinique des réponses libres relève d'une
relecture humaine ; toutes les prédictions sont conservées à cet effet.

## Limites et risques

- **Connaissances médicales** : un modèle de 1,7 milliard de paramètres apprend surtout le format et
  le style. Ses réponses peuvent être fausses tout en paraissant assurées.
- **Français en réponse libre** : non appris, faute de question ouverte française dans le train.
  Une petite part des réponses aux cas cliniques français est rédigée en anglais.
- **Boucles de répétition résiduelles** : environ une réponse ouverte sur six atteint encore la
  limite de 512 tokens, même avec la pénalité de répétition.
- **QCM multiples** : tendance à cocher trop d'options.
- **Jeu de test** : 3 quasi-doublons de questions FrenchMedMCQA du train (reformulations
  d'annales), d'effet négligeable sur les scores ; ils révèlent une limite de la déduplication par
  correspondance exacte.
- **Contamination possible** : FrenchMedMCQA est public depuis 2022 et a pu figurer dans le
  pré-entraînement de Qwen. Le protocole ne permet pas de le tester.
- **Triage** : non appris, non évalué.

## Utilisation

Le dépôt est **privé** : un jeton Hugging Face autorisé est nécessaire. Il est recommandé de
**fusionner** l'adaptateur avant l'inférence, comme dans l'évaluation, et de servir le modèle
fusionné, notamment avec vLLM.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id, adapter_id = "Qwen/Qwen3-1.7B-Base", "saidmo2/chsa-qwen3-1.7b-sft-lora"
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload().eval()

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

Points importants :
- `enable_thinking=False` : le prompt doit se terminer par le bloc `<think>` vide, comme à
  l'entraînement ;
- l'arrêt se fait sur `<|im_end|>`, et non sur `<|endoftext|>` comme pour le modèle de base ;
- `repetition_penalty=1.1` pour les réponses libres ; inutile pour les QCM.

## Traçabilité et vérifications

- **Révisions** : modèle de base, dataset et commit Git relevés au lancement et tracés dans
  MLflow, avec la configuration et le `pip freeze` de l'environnement.
- **Adaptateur évalué** : révision `a7c6bf2d639ae2ba1c39ae18c266c4107e897462`, dont les 393
  tenseurs sont identiques bit à bit à ceux du meilleur checkpoint (pas 550).
- **Fusion** : les vecteurs appris des quatre tokens sont conservés à l'identique par
  `merge_and_unload`, et l'entrée et la sortie restent liées.

## Citations

- Bazoge A. (2025). *MediQAl*. arXiv:2507.20917.
- Labrak Y. et al. (2022). *FrenchMedMCQA*. LOUHI 2022.
- Ben Abacha A., Demner-Fushman D. (2019). *A question-entailment approach to question answering*.
  BMC Bioinformatics 20, 511.
- Qwen Team (2025). *Qwen3*.
