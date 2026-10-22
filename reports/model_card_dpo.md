---
base_model: saidmo2/chsa-qwen3-1.7b-sft
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
  - fr
  - en
datasets:
  - saidmo2/chsa-triage-medical
tags:
  - dpo
  - lora
  - trl
  - peft
  - medical
  - french
  - proof-of-concept
---

# chsa-qwen3-1.7b-sft-dpo-lora — adaptateur DPO (second modèle du POC CHSA)

Adaptateur **LoRA** entraîné par **alignement par préférences (DPO)** sur le modèle SFT fusionné
[`saidmo2/chsa-qwen3-1.7b-sft`](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft), dans le cadre du
POC d'assistant médical du **Centre Hospitalier Saint-Aurélien (CHSA)**, client fictif du projet 14 du
parcours AI Engineer d'OpenClassrooms.

**Statut : servi comme second modèle, pas par défaut.** Il apprend les préférences visées sans
dégrader les QCM, mais n'a pas démontré être au moins aussi sûr que le modèle SFT, comme l'exigeaient
les critères d'acceptation fixés avant son évaluation. Le modèle servi par défaut est le SFT fusionné.

> ⚠️ **Modèle de démonstration, inutilisable cliniquement.** Sur 20 cas cliniques d'urgence et de
> réanimation relus, aucune réponse n'est juste et 8 sont jugées dangereuses (6 pour le SFT). Il ne
> doit pas être utilisé pour conseiller un patient, orienter une prise en charge ni attribuer un
> niveau de priorité de triage.

La **carte complète** (données, entraînement, critères, évaluation, relecture de sécurité,
diagnostic de fusion, limites, traçabilité) est celle du
[modèle SFT fusionné](https://huggingface.co/saidmo2/chsa-qwen3-1.7b-sft).

## Entraînement

| Élément | Valeur |
|---|---|
| Modèle de départ et de référence | `saidmo2/chsa-qwen3-1.7b-sft`, révision `923746e75cca6091cbadae42a7a052addc4bb2db` |
| Données | `saidmo2/chsa-triage-medical`, config `dpo`, révision `ba1522abfacbe127a54fae2b7cdb35d0de14fa89` : 3 000 paires (UltraMedical-Preference : ChatDoctor, MedQA, MedQA-Evol), en anglais, jugées par GPT-4 |
| Méthode | DPO standard (sigmoid), β 0,1, TRL 1.14.1, PEFT 0.21.2 |
| LoRA | r 16, alpha 32, toutes les couches linéaires (identique au SFT) ; tokens du chat template gelés |
| Optimisation | 1 époque (188 pas), lr 5e-6 cosinus, warmup 19 pas, batch effectif 16, bf16, dropout désactivé |
| Sélection | meilleur checkpoint sur la loss DPO de validation : pas 175 (0,494) |
| Run | `dpo-20261013-154008` (MLflow `chsa-dpo`), commit `0c5cd51633ac8bcadb06791c08d2bc7e3bc3e54e`, A40, ≈ 0,41 $ |

## Résultats clés

| Indicateur | SFT | Cet adaptateur |
|---|---|---|
| Précision des récompenses, test DPO (n = 777) | — | **75,4 %** (73,6 % sur les 163 paires annotées par des humains) |
| Format QCM respecté | 100 % | 100 % |
| FrenchMedMCQA, exactitude | 57,9 % | 56,1 % |
| Réponses tronquées, cas cliniques / MedQuAD | 17,1 % / 6,7 % | 20,2 % / 6,3 % |
| Réponses dangereuses (relecture, 20 cas) | 6 | 8 |

Les écarts de troncature et de réponses dangereuses ne sont pas significatifs (McNemar p = 0,32,
test du signe p = 0,73). Le détail et les limites de la relecture sont dans la carte complète.

## Ne pas fusionner

La mise à jour DPO est si petite (0,024 % de la norme des poids) que la fusion en bf16 efface 94 %
de ses coefficients ; la fusion a été refusée par les contrôles du projet. L'adaptateur doit être
**appliqué à l'exécution**, avec PEFT ou avec le support LoRA de vLLM.

## Utilisation

Les dépôts sont **privés** : un jeton Hugging Face autorisé est nécessaire.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, base_rev = "saidmo2/chsa-qwen3-1.7b-sft", "923746e75cca6091cbadae42a7a052addc4bb2db"
adapter, adapter_rev = "saidmo2/chsa-qwen3-1.7b-sft-dpo-lora", "8818097df7fab610d9b8109bcdc435192b893199"

tokenizer = AutoTokenizer.from_pretrained(base, revision=base_rev)
model = AutoModelForCausalLM.from_pretrained(base, revision=base_rev, dtype=torch.bfloat16,
                                             device_map="auto")
model = PeftModel.from_pretrained(model, adapter, revision=adapter_rev).eval()   # sans fusion
```

La génération est identique à celle du modèle SFT (voir la carte complète) : `enable_thinking=False`
dans le chat template, arrêt sur `<|im_end|>`, `repetition_penalty=1.1` pour les réponses libres.

La révision `8818097df7fab610d9b8109bcdc435192b893199` est celle des poids entraînés et évalués ; publier cette carte crée une
nouvelle révision du dépôt sans changer les poids.
