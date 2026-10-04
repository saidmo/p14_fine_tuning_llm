# Inventaire des sources — 2026-10-01T15:10:08+00:00

## mediqal (`ANR-MALADES/MediQAl`)

- Langue : fr — rôle prévu : sft
- Licence (card) : cc-by-4.0 — révision : `5af34948a74c7b8807c476204a21149ffb00ea2c`

### Config `mcqu`

**train** — 10113 lignes

Colonnes : `id`, `clinical_case`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `task`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 5 / 5, `clinical_case` 592 / 1651, `question` 99 / 772, `answer_a` 23 / 141, `answer_b` 24 / 139, `answer_c` 24 / 141, `answer_d` 24 / 146, `answer_e` 25 / 147, `correct_answers` 1 / 1, `task` 3 / 3, `medical_subject` 12 / 32, `question_type` 13 / 13

```json
{
  "id": "7543",
  "clinical_case": "Monsieur R. âgé de 37 ans, forestier de son métier, doit partir travailler pendant 2 ans au Gabon dans la région de Lambaréné. Monsieur R. a l'intention de se faire accompagner de son épouse âgée de 36 ans et de son fils âgé de 5 ans. Ils logeront à quelques kilomètres de la forêt mais Mr R., de par…",
  "question": "Au sujet des vaccinations :",
  "answer_a": "Le vaccin contre la fièvre jaune est obligatoire",
  "answer_b": "Le vaccin contre la fièvre jaune peut se faire au cabinet du médecin traitant",
  "answer_c": "Le vaccin contre le choléra est un vaccin qui assure une protection supérieure à 80%",
  "answer_d": "Le vaccin contre l'hépatite B est conseillé mais il ne se trouve pas en pharmacie, il faut contacter l'hôpital",
  "answer_e": "Les vaccinations contre le tétanos et la poliomyélite ne sont pas indispensables",
  "correct_answers": "A",
  "task": "QCU",
  "medical_subject": "Infectious Diseases",
  "question_type": "Understanding"
}
```

**validation** — 2561 lignes

Colonnes : `id`, `clinical_case`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `task`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 5 / 5, `clinical_case` 573 / 1609, `question` 105 / 533, `answer_a` 23 / 136, `answer_b` 23 / 132, `answer_c` 24 / 144, `answer_d` 24 / 129, `answer_e` 24 / 138, `correct_answers` 1 / 1, `task` 3 / 3, `medical_subject` 12 / 32, `question_type` 13 / 13

```json
{
  "id": "9940",
  "clinical_case": null,
  "question": "L'amblyopie fonctionnelle se définit comme :",
  "answer_a": "Une déviation des axes oculaires l'un par rapport à l'autre",
  "answer_b": "Une paralysie de la motricité pupillaire",
  "answer_c": "Une faible acuité visuelle sans cause organique décelable",
  "answer_d": "Une perte du champ visuel périphérique",
  "answer_e": "Aucune des définitions citées",
  "correct_answers": "C",
  "task": "QCU",
  "medical_subject": "Ophthalmology",
  "question_type": "Understanding"
}
```

**test** — 4343 lignes

Colonnes : `id`, `clinical_case`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `task`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 5 / 5, `clinical_case` 571 / 2487, `question` 104 / 827, `answer_a` 22 / 234, `answer_b` 23 / 222, `answer_c` 24 / 255, `answer_d` 24 / 252, `answer_e` 25 / 233, `correct_answers` 1 / 1, `task` 3 / 3, `medical_subject` 11 / 32, `question_type` 13 / 13

```json
{
  "id": "23648",
  "clinical_case": "Un patient âgé de 65 ans, présente depuis 2 ans, une pollakiurie diurne et nocturne avec des épisodes d’hématurie terminale.\r\nL'examen clinique met en évidence une masse lombaire donnant le contact lombaire à gauche.\r\nLe toucher rectal retrouve un adénome de la prostate d'environ 80 gr. L'échographi…",
  "question": "(cochez la réponse juste) Il est logique de :",
  "answer_a": "Rechercher les métastases",
  "answer_b": "Entamer une antibiothérapie adaptée plus cathéter sus pubien",
  "answer_c": "Antibiothérapie à large spectre plus cathéter trans uréthral",
  "answer_d": "Cathéter sus pubien plus antibiothérapie adaptée plus bilan d'extension et néphrectomie",
  "answer_e": "Le malade a consulté pour des signes du bas appareil urinaire, il vaut mieux commencer par le traitement de la prostate avant de s'occuper du cancer du rein.",
  "correct_answers": "A",
  "task": "QCU",
  "medical_subject": "Nephro-Urology",
  "question_type": "Reasoning"
}
```

### Config `mcqm`

**train** — 5767 lignes

Colonnes : `id`, `clinical_case`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `task`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 5 / 5, `clinical_case` 669 / 2487, `question` 88 / 511, `answer_a` 25 / 121, `answer_b` 26 / 137, `answer_c` 27 / 135, `answer_d` 28 / 150, `answer_e` 28 / 148, `correct_answers` 5 / 9, `task` 3 / 3, `medical_subject` 12 / 32, `question_type` 13 / 13

```json
{
  "id": "11399",
  "clinical_case": "Plusieurs jours après un contage avec un enfant atteint d'oreillons, une fillette de 6 ans développe un état fébrile à 40° rapidement suivi d'une parotidite unilatérale. En 48h, l'atteinte salivaire se bilatéralise et la température se normalise. Vers le 8e jour cependant, apparaissent des céphalées…",
  "question": "Quelle(s) proposition(s) peut (peuvent) s'appliquer à l'épidémiologie de la maladie ?",
  "answer_a": "Transmission manuportée",
  "answer_b": "La phase de contagiosité dure 8 jours",
  "answer_c": "Maladie strictement humaine",
  "answer_d": "Confère une immunité",
  "answer_e": "Transmission indirecte possible",
  "correct_answers": "C,D",
  "task": "QCM",
  "medical_subject": "Infectious Diseases",
  "question_type": "Understanding"
}
```

**validation** — 1466 lignes

Colonnes : `id`, `clinical_case`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `task`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 5 / 5, `clinical_case` 622 / 1846, `question` 88 / 405, `answer_a` 27 / 150, `answer_b` 27 / 155, `answer_c` 29 / 133, `answer_d` 28 / 140, `answer_e` 28 / 140, `correct_answers` 5 / 9, `task` 3 / 3, `medical_subject` 12 / 32, `question_type` 13 / 13

```json
{
  "id": "19099",
  "clinical_case": "Un patient agé de 76 ans se présente avec un tableau d'insuffisance cardiaque globale décompensée. Malgré des doses élevées de diurétiques, 750 mg de furosémide et la prescription d'un inhibiteur de l'enzyme de conversion, il persiste des oedèmes périphériques, une congestion pulmonaire, une oliguri…",
  "question": "Les actions pharmacodynamiques de la Dobutamine sont:",
  "answer_a": "Bronchodilatation à faibles doses",
  "answer_b": "Tachycardie à fortes doses",
  "answer_c": "Stimulation inotrope dose dépendante",
  "answer_d": "Action bathmotrope dose dépendante",
  "answer_e": "Vasodilatation artérielle périphérique à faible dose",
  "correct_answers": "B,C,D",
  "task": "QCM",
  "medical_subject": "Pharmacology",
  "question_type": "Understanding"
}
```

**test** — 3384 lignes

Colonnes : `id`, `clinical_case`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `task`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 5 / 5, `clinical_case` 684 / 2187, `question` 90 / 360, `answer_a` 28 / 147, `answer_b` 27 / 146, `answer_c` 28 / 140, `answer_d` 28 / 146, `answer_e` 28 / 154, `correct_answers` 3 / 9, `task` 3 / 3, `medical_subject` 12 / 32, `question_type` 13 / 13

```json
{
  "id": "21347",
  "clinical_case": "Un homme de 52 ans, éthylique chronique, consulte pour une diarrhée survenue il y a quelques mois dans les suites d'une intervention pour un adénocarcinome lieberkühnien du côlon qui a enlevé le côlon droit et 20 cm d'iléon. Il s'agissait d'un cancer au stade C de Dukes. Il n'y a pas de douleurs et …",
  "question": "Parmi les maladies suivantes, quelle(s) est(sont) la(les) cause(s) de diarrhée motrice ?",
  "answer_a": "Hyperthyroïdie",
  "answer_b": "Tumeur carcinoïde entraînant un syndrome carcinoïde",
  "answer_c": "Cancer médullaire du corps thyroïde",
  "answer_d": "Déconjugaision des sels biliaires dans l'iléon",
  "answer_e": "Diabète avec neuropathie végétative",
  "correct_answers": "A,B,C,D,E",
  "task": "QCM",
  "medical_subject": "Hepato-Gastroenterology",
  "question_type": "Understanding"
}
```

### Config `oeq`

**test** — 4969 lignes

Colonnes : `id`, `clinical_case`, `cc_question_number`, `question`, `answer`, `medical_subject`, `question_type`

Longueurs (car., médiane / max) : `id` 4 / 4, `clinical_case` 732 / 3967, `cc_question_number` 1 / 2, `question` 99 / 1545, `answer` 176 / 991, `medical_subject` 11 / 32, `question_type` 9 / 13

```json
{
  "id": "1",
  "clinical_case": "Homme, 58 ans, majoration de dyspnée chez un BPCO connu depuis 20 ans.\n\n Examen d’entrée : \n Constantes :\n - FC = 130bpm\n - PA = 130/80 aux deux bras\n - FR = 40/min\n - SpO2 = 80%\n - T = 38 ,5°C\n Tirage sus-sternal\n Cyanose\n Sueurs\n Auscultation : \n - Ronchi bilatéraux et sibilants\n - Tympanisme bila…",
  "cc_question_number": "1",
  "question": "Quels sont éléments de gravité au moment de l’arrivée du patient ?",
  "answer": "FC>125bpm\n Terrain : ATCD de BPCO, insuffisance respiratoire chronique (hippocratisme digital)\n Signes de détresse respiratoire avec hypoxie et hypercapnie :\n - SpO2<90%\n - FR>30/min\n - Sueurs\n - Cyanose\n - Tirage\n Signes d’insuffisance droite\n - Hépatomégalie douloureuse\n - Reflux\n - Turgescence",
  "medical_subject": "Emergency Medicine",
  "question_type": "Reasoning"
}
```

## frenchmedmcqa (`nthngdy/frenchmedmcqa`)

- Langue : fr — rôle prévu : sft
- Licence (card) : None — révision : `6195120803580d171cbf2172c0beac03f21d14fb`

### Config `default`

**train** — 595 lignes

Colonnes : `id`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `number_correct_answers`

Longueurs (car., médiane / max) : `id` 64 / 64, `question` 92 / 491, `answer_a` 28 / 146, `answer_b` 30 / 173, `answer_c` 32 / 186, `answer_d` 32 / 156, `answer_e` 35 / 222

```json
{
  "id": "230bac49b0fe863b772410bc8d01a025f63c3c999065480131d6334abd2efeff",
  "question": "Parmi les affirmations suivantes, une seule est fausse, indiquer laquelle: les particules alpha",
  "answer_a": "Sont formées de noyaux d'hélium",
  "answer_b": "Sont peu pénétrantes",
  "answer_c": "Toute l'énergie qu'elles transportent est cédée au long d'un parcours de quelques centimètres dans l'air",
  "answer_d": "Sont arrêtées par une feuille de papier",
  "answer_e": "Sont peu ionisantes",
  "correct_answers": 4,
  "number_correct_answers": 0
}
```

**validation** — 164 lignes

Colonnes : `id`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `number_correct_answers`

Longueurs (car., médiane / max) : `id` 64 / 64, `question` 100 / 323, `answer_a` 26 / 116, `answer_b` 29 / 121, `answer_c` 28 / 135, `answer_d` 29 / 131, `answer_e` 28 / 150

```json
{
  "id": "d0f974df4406bf2f933962bf0d0c0447a80d66719031e3b4a35c2ff3b3ccc594",
  "question": "L'adrénaline stimule tous ces processus métaboliques sauf un. Lequel?",
  "answer_a": "Glycolyse",
  "answer_b": "Lipolyse",
  "answer_c": "Néoglucogénèse",
  "answer_d": "Glycogénolyse",
  "answer_e": "Cétogénèse",
  "correct_answers": 4,
  "number_correct_answers": 0
}
```

**test** — 321 lignes

Colonnes : `id`, `question`, `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e`, `correct_answers`, `number_correct_answers`

Longueurs (car., médiane / max) : `id` 64 / 64, `question` 105 / 359, `answer_a` 30 / 122, `answer_b` 33 / 134, `answer_c` 32 / 133, `answer_d` 31 / 122, `answer_e` 33 / 132

```json
{
  "id": "5987fa6bffd499eb439c90679d7fbca822d62bc639d1b9c94c68ae20e46f6004",
  "question": "Parmi les propositions suivantes, indiquer celle qui est exacte. Dans les conditions physiologiques, le pH le plus élevé est mesuré dans:",
  "answer_a": "Le suc gastrique",
  "answer_b": "La bile vésiculaire",
  "answer_c": "Le suc pancréatique",
  "answer_d": "La salive",
  "answer_e": "Les sécrétions intestinales",
  "correct_answers": 2,
  "number_correct_answers": 0
}
```

## medquad (`keivalya/MedQuad-MedicalQnADataset`)

- Langue : en — rôle prévu : sft
- Licence (card) : None — révision : `5b0961fbaa6d7f9c344c5d59c29943fb900c2eca`

### Config `default`

**train** — 16407 lignes

Colonnes : `qtype`, `Question`, `Answer`

Longueurs (car., médiane / max) : `qtype` 11 / 15, `Question` 48 / 166, `Answer` 902 / 23065

```json
{
  "qtype": "susceptibility",
  "Question": "Who is at risk for Lymphocytic Choriomeningitis (LCM)? ?",
  "Answer": "LCMV infections can occur after exposure to fresh urine, droppings, saliva, or nesting materials from infected rodents.  Transmission may also occur when these materials are directly introduced into broken skin, the nose, the eyes, or the mouth, or presumably, via the bite of an infected rodent. Per…"
}
```

## ultramedical_preference (`TsinghuaC3I/UltraMedical-Preference`)

- Langue : en — rôle prévu : dpo
- Licence (card) : mit — révision : `761eb7935310ba662a96d93c5af342e5269d5759`

### Config `default`

**train** — 109353 lignes

Colonnes : `prompt_id`, `label_type`, `prompt`, `chosen`, `rejected`, `metadata`, `feedback`

Longueurs (car., médiane / max) : `prompt_id` 17 / 39, `label_type` 4 / 6, `prompt` 399 / 3771, `feedback` 1719 / 3047

```json
{
  "prompt_id": "WikiInstruct,8304",
  "label_type": "length",
  "prompt": "Investigate the intricacies of immunometabolism, a distinct subfield of immunology that examines the interconnection between cellular metabolic processes and the functional attributes of immune cells. Clarify the mechanisms by which this symbiosis modulates the comprehensive immune response, and ana…",
  "chosen": [
    {
      "content": "Investigate the intricacies of immunometabolism, a distinct subfield of immunology that examines the interconnection between cellular metabolic processes and the functional attributes of immune cells. Clarify the mechanisms by which this symbiosis modulates the comprehensive immune response, and ana…",
      "role": "user"
    },
    {
      "content": " Immunometabolism is an emerging field of study that explores the intersection between cellular metabolic processes and the functional attributes of immune cells. This subfield has gained significant attention due to the increasing recognition that immune cell functions are heavily influenced by the…",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "Investigate the intricacies of immunometabolism, a distinct subfield of immunology that examines the interconnection between cellular metabolic processes and the functional attributes of immune cells. Clarify the mechanisms by which this symbiosis modulates the comprehensive immune response, and ana…",
      "role": "user"
    },
    {
      "content": "Immunometabolism is an emerging field that explores how metabolic processes influence the function and fate of immune cells, and vice versa, how immune responses can shape cellular metabolism. This field has been gaining attention due to its implications in a variety of diseases, including infection…",
      "role": "assistant"
    }
  ],
  "metadata": {
    "golden_answer": null,
    "chosen": {
      "model": "Mixtral-8x7B-Instruct",
      "score": 5.0,
      "rank": 1,
      "evaluation": "This response is exceptionally detailed and well-structured, providing an in-depth exploration of the mechanisms of immunometabolism and its implications. It effectively integrates immunometabolism within the broader context of immunological studies and discusses its impact on metabolic diseases ext…"
    },
    "rejected": {
      "model": "Llama-3-8B-UltraMedical",
      "score": 4.0,
      "rank": 2,
      "evaluation": "The response provides a comprehensive overview of immunometabolism, detailing the mechanisms and implications for metabolic diseases. It effectively addresses the scientific inquiry with appropriate terminology and clinical relevance, demonstrating a strong grasp of the subject. However, it could im…"
    }
  },
  "feedback": "Assistant A provides a detailed and comprehensive response that thoroughly explains the concept of immunometabolism, the mechanisms involved, and its implications in various diseases, particularly metabolic disorders. The answer includes specific examples of how different immune cells undergo metabo…"
}
```

**validation** — 2232 lignes

Colonnes : `prompt_id`, `label_type`, `prompt`, `chosen`, `rejected`, `metadata`, `feedback`

Longueurs (car., médiane / max) : `prompt_id` 17 / 39, `label_type` 4 / 6, `prompt` 411 / 2098, `feedback` 1697 / 3303

```json
{
  "prompt_id": "MedQA,4303",
  "label_type": "hard",
  "prompt": "A 62-year-old Caucasian male presents to your office with hemoptysis and hematuria. On physical exam you note a saddle nose deformity. Laboratory results show an elevated level of cytoplasmic antineutrophil cytoplasmic antibody. Which of the following interventions is most appropriate for this patie…",
  "chosen": [
    {
      "content": "A 62-year-old Caucasian male presents to your office with hemoptysis and hematuria. On physical exam you note a saddle nose deformity. Laboratory results show an elevated level of cytoplasmic antineutrophil cytoplasmic antibody. Which of the following interventions is most appropriate for this patie…",
      "role": "user"
    },
    {
      "content": "The patient presents with hemoptysis (coughing up blood), hematuria (blood in the urine), and a saddle nose deformity, which is a collapse of the nasal bridge. The presence of an elevated level of cytoplasmic antineutrophil cytoplasmic antibody (c-ANCA) strongly suggests a diagnosis of Granulomatosi…",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "A 62-year-old Caucasian male presents to your office with hemoptysis and hematuria. On physical exam you note a saddle nose deformity. Laboratory results show an elevated level of cytoplasmic antineutrophil cytoplasmic antibody. Which of the following interventions is most appropriate for this patie…",
      "role": "user"
    },
    {
      "content": "\n\nLet's break down the patient's symptoms and lab results:\n\n* Hemoptysis and hematuria suggest a bleeding disorder or vasculitis.\n* A saddle nose deformity is a characteristic feature of granulomatosis with polyangiitis (GPA), a type of vasculitis.\n* Elevated cytoplasmic antineutrophil cytoplasmic a…",
      "role": "assistant"
    }
  ],
  "metadata": {
    "golden_answer": "C",
    "chosen": {
      "model": "gpt-4-1106-preview",
      "score": 5.0,
      "rank": 2,
      "evaluation": "The response is detailed, explaining the symptoms, diagnosis, and rationale for the treatment choice. It discusses other options and why they are less appropriate, providing a comprehensive clinical explanation."
    },
    "rejected": {
      "model": "Meta-Llama-3-70B-Instruct",
      "score": 4.0,
      "rank": 7,
      "evaluation": "The response clearly identifies the condition and appropriate treatment. It succinctly explains why corticosteroids are necessary and briefly touches on why other options are not suitable."
    }
  },
  "feedback": "Both Assistant A and Assistant B provided responses that correctly identified corticosteroids as the most appropriate intervention for the patient described, aligning with the reference answer (C). Both assistants recognized the symptoms and laboratory findings as indicative of Granulomatosis with p…"
}
```

**test** — 777 lignes

Colonnes : `prompt_id`, `label_type`, `prompt`, `chosen`, `rejected`, `metadata`, `feedback`

Longueurs (car., médiane / max) : `prompt_id` 17 / 37, `label_type` 4 / 6, `prompt` 399 / 2532

```json
{
  "prompt_id": "ChatDoctor,19683",
  "label_type": "human",
  "prompt": "Hi; I am a 51 yo female; 5 3 130 lbs. I started having sharp, pulsating left sided chest pain this morning while folding laundry. Lots of pressure, like I had something heavy sitting on my chest. The pain went through to my spine. I laid down to rest and fell asleep; when I woke, the pain was mostly…",
  "chosen": [
    {
      "content": "Hi; I am a 51 yo female; 5 3 130 lbs. I started having sharp, pulsating left sided chest pain this morning while folding laundry. Lots of pressure, like I had something heavy sitting on my chest. The pain went through to my spine. I laid down to rest and fell asleep; when I woke, the pain was mostly…",
      "role": "user"
    },
    {
      "content": "I'm so sorry to hear that you're experiencing chest pain. While it's always a good idea to consult with a healthcare professional for a proper evaluation and diagnosis, I can try to help you better understand your symptoms and provide some possible explanations.\n\nBased on your description, it sounds…",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "Hi; I am a 51 yo female; 5 3 130 lbs. I started having sharp, pulsating left sided chest pain this morning while folding laundry. Lots of pressure, like I had something heavy sitting on my chest. The pain went through to my spine. I laid down to rest and fell asleep; when I woke, the pain was mostly…",
      "role": "user"
    },
    {
      "content": "\n\nI'm not a doctor, but I can try to provide some general information and guidance. It's always best to consult a healthcare professional for personalized advice.\n\nBased on your symptoms, it's essential to consider the possibility of a cardiovascular event, such as a heart attack or angina. The shar…",
      "role": "assistant"
    }
  ],
  "metadata": {
    "golden_answer": "Thanks for your question on Chat Doctor. I can understand your situation and problem. By your symptoms and description, possibility of cardiac cause is more in your case. So better to consult cardiologist and get done 1. Clinical examination 2. ECG 3. 2d echo 4. Stress test. You are mostly having ca…",
    "chosen": {
      "model": "Meta-Llama-3-8B-Instruct",
      "score": 2.0,
      "rank": 8,
      "evaluation": "This response misinterprets the severity of the symptoms by suggesting they are more likely musculoskeletal rather than cardiac. While it provides a detailed exploration of possible non-cardiac causes, it dangerously downplays the need for urgent medical evaluation for potential heart-related issues…"
    },
    "rejected": {
      "model": "Meta-Llama-3-70B-Instruct",
      "score": 4.0,
      "rank": 2,
      "evaluation": "The response appropriately suggests that the symptoms could be indicative of a cardiovascular event and advises immediate medical attention. It also lists other potential causes, providing a broad perspective. However, the response could be more assertive in urging immediate emergency care, consider…"
    }
  },
  "feedback": null
}
```
