# Profilage des sources — 2026-10-01T15:49:28+00:00

Tokenizer : `Qwen/Qwen3-1.7B-Base` — seed : 42

## mediqal

```json
{
  "mcqu": {
    "train": {
      "rows": 10113,
      "with_clinical_case": 3935,
      "question_type": {
        "Understanding": 6933,
        "Reasoning": 3180
      },
      "medical_subject": {
        "Cardiology": 691,
        "Hepato-Gastroenterology": 661,
        "Gynecology and Obstetrics": 626,
        "Endocrinology and Metabolism": 552,
        "Pulmonology": 535,
        "Hematology": 525,
        "Nephro-Urology": 496,
        "Pediatrics": 467,
        "Psychiatry": 419,
        "Rheumatology": 394,
        "Neurology": 394,
        "Infectious Diseases": 390,
        "Otorhinolaryngology (ENT)": 333,
        "Orthopedics": 323,
        "Pathological Anatomy": 308,
        "Immunology": 297,
        "Ophthalmology": 295,
        "Biochemistry": 291,
        "Dermatology": 289,
        "Microbiology": 271,
        "Epidemiology": 268,
        "Forensic Medicine and Toxicology": 200,
        "Occupational Medicine": 198,
        "Pharmacology": 193,
        "Physiology": 169,
        "Parasitology": 167,
        "Genetics": 165,
        "Histology": 110,
        "Radiology": 35,
        "Anatomy": 28,
        "Cytology": 14,
        "Rehabilitation": 5,
        "Embryology": 4
      }
    },
    "validation": {
      "rows": 2561,
      "with_clinical_case": 999,
      "question_type": {
        "Understanding": 1749,
        "Reasoning": 812
      },
      "medical_subject": {
        "Cardiology": 174,
        "Hepato-Gastroenterology": 167,
        "Gynecology and Obstetrics": 157,
        "Endocrinology and Metabolism": 139,
        "Pulmonology": 134,
        "Hematology": 132,
        "Nephro-Urology": 125,
        "Pediatrics": 118,
        "Psychiatry": 106,
        "Rheumatology": 100,
        "Infectious Diseases": 99,
        "Neurology": 99,
        "Otorhinolaryngology (ENT)": 84,
        "Orthopedics": 82,
        "Pathological Anatomy": 78,
        "Immunology": 76,
        "Ophthalmology": 74,
        "Biochemistry": 74,
        "Dermatology": 73,
        "Microbiology": 69,
        "Epidemiology": 68,
        "Forensic Medicine and Toxicology": 51,
        "Occupational Medicine": 50,
        "Pharmacology": 50,
        "Physiology": 44,
        "Parasitology": 43,
        "Genetics": 42,
        "Histology": 28,
        "Radiology": 10,
        "Anatomy": 8,
        "Cytology": 4,
        "Rehabilitation": 2,
        "Embryology": 1
      }
    },
    "test": {
      "rows": 4343,
      "with_clinical_case": 1996,
      "question_type": {
        "Reasoning": 1689,
        "Understanding": 2654
      },
      "medical_subject": {
        "Cardiology": 327,
        "Hematology": 317,
        "Gynecology and Obstetrics": 280,
        "Hepato-Gastroenterology": 278,
        "Pulmonology": 248,
        "Nephro-Urology": 227,
        "Pediatrics": 215,
        "Rheumatology": 209,
        "Endocrinology and Metabolism": 207,
        "Infectious Diseases": 192,
        "Neurology": 177,
        "Orthopedics": 172,
        "Psychiatry": 142,
        "Otorhinolaryngology (ENT)": 127,
        "Dermatology": 125,
        "Occupational Medicine": 107,
        "Biochemistry": 103,
        "Forensic Medicine and Toxicology": 99,
        "Epidemiology": 97,
        "Ophthalmology": 95,
        "Immunology": 91,
        "Pathological Anatomy": 78,
        "Genetics": 72,
        "Histology": 71,
        "Physiology": 69,
        "Microbiology": 63,
        "Pharmacology": 54,
        "Parasitology": 46,
        "Radiology": 28,
        "Anatomy": 18,
        "Cytology": 6,
        "Embryology": 3
      }
    }
  },
  "mcqm": {
    "train": {
      "rows": 5767,
      "with_clinical_case": 2185,
      "question_type": {
        "Understanding": 4320,
        "Reasoning": 1447
      },
      "medical_subject": {
        "Cardiology": 395,
        "Hepato-Gastroenterology": 395,
        "Pulmonology": 341,
        "Psychiatry": 329,
        "Nephro-Urology": 328,
        "Endocrinology and Metabolism": 300,
        "Infectious Diseases": 291,
        "Gynecology and Obstetrics": 290,
        "Rheumatology": 289,
        "Hematology": 281,
        "Neurology": 278,
        "Dermatology": 270,
        "Pediatrics": 249,
        "Otorhinolaryngology (ENT)": 220,
        "Ophthalmology": 196,
        "Forensic Medicine and Toxicology": 172,
        "Immunology": 158,
        "Pharmacology": 147,
        "Biochemistry": 142,
        "Pathological Anatomy": 141,
        "Orthopedics": 127,
        "Microbiology": 100,
        "Epidemiology": 97,
        "Occupational Medicine": 93,
        "Genetics": 90,
        "Parasitology": 43,
        "Rehabilitation": 5
      }
    },
    "validation": {
      "rows": 1466,
      "with_clinical_case": 547,
      "question_type": {
        "Understanding": 1092,
        "Reasoning": 374
      },
      "medical_subject": {
        "Hepato-Gastroenterology": 100,
        "Cardiology": 100,
        "Pulmonology": 86,
        "Nephro-Urology": 84,
        "Psychiatry": 84,
        "Endocrinology and Metabolism": 75,
        "Infectious Diseases": 74,
        "Gynecology and Obstetrics": 73,
        "Rheumatology": 73,
        "Neurology": 71,
        "Hematology": 71,
        "Dermatology": 68,
        "Pediatrics": 64,
        "Otorhinolaryngology (ENT)": 56,
        "Ophthalmology": 49,
        "Forensic Medicine and Toxicology": 44,
        "Immunology": 40,
        "Pharmacology": 37,
        "Biochemistry": 37,
        "Pathological Anatomy": 36,
        "Orthopedics": 32,
        "Microbiology": 26,
        "Epidemiology": 25,
        "Occupational Medicine": 24,
        "Genetics": 23,
        "Parasitology": 12,
        "Rehabilitation": 2
      }
    },
    "test": {
      "rows": 3384,
      "with_clinical_case": 1342,
      "question_type": {
        "Understanding": 2330,
        "Reasoning": 1054
      },
      "medical_subject": {
        "Cardiology": 237,
        "Hepato-Gastroenterology": 230,
        "Hematology": 216,
        "Nephro-Urology": 213,
        "Pulmonology": 211,
        "Gynecology and Obstetrics": 185,
        "Rheumatology": 183,
        "Endocrinology and Metabolism": 180,
        "Psychiatry": 165,
        "Dermatology": 159,
        "Neurology": 158,
        "Pediatrics": 145,
        "Infectious Diseases": 140,
        "Otorhinolaryngology (ENT)": 137,
        "Ophthalmology": 130,
        "Forensic Medicine and Toxicology": 105,
        "Orthopedics": 103,
        "Immunology": 67,
        "Genetics": 67,
        "Pharmacology": 65,
        "Epidemiology": 65,
        "Occupational Medicine": 60,
        "Pathological Anatomy": 53,
        "Biochemistry": 45,
        "Microbiology": 38,
        "Parasitology": 25,
        "Rehabilitation": 2
      }
    }
  },
  "oeq": {
    "test": {
      "rows": 4969,
      "with_clinical_case": 3954,
      "question_type": {
        "Reasoning": 3125,
        "Understanding": 1842,
        "Raisonnement": 2
      },
      "medical_subject": {
        "Pharmacy": 1544,
        "Pediatric Cardiology": 1225,
        "Cardiology": 217,
        "Nephro-Urology": 193,
        "Pulmonology": 184,
        "Intensive Care": 154,
        "Neurology": 150,
        "Gynecology and Obstetrics": 149,
        "Otorhinolaryngology (ENT)": 126,
        "Hematology": 124,
        "Emergency Medicine": 109,
        "Orthopedics": 93,
        "Infectious Diseases": 88,
        "Forensic Medicine and Toxicology": 86,
        "Psychiatry": 80,
        "Hepato-Gastroenterology": 68,
        "Genetics": 65,
        "Dermatology": 54,
        "Pediatrics": 47,
        "Rheumatology": 37,
        "Ophthalmology": 31,
        "Endocrinology and Metabolism": 24,
        "Parasitology": 23,
        "Public Health": 19,
        "Bacteriology": 17,
        "Immunology": 16,
        "Occupational Medicine": 16,
        "Hématologie": 13,
        "Pharmacology": 9,
        "Rehabilitation": 6,
        "Semiology": 2
      }
    }
  }
}
```

## frenchmedmcqa

```json
{
  "train": {
    "rows": 595,
    "features": {
      "correct_answers": "Value('int64')",
      "number_correct_answers": "ClassLabel(names=['1', '2', '3', '4', '5'])"
    },
    "number_correct_answers": {
      "0": 595
    },
    "correct_answers": {
      "4": 84,
      "2": 142,
      "1": 119,
      "3": 156,
      "0": 94
    }
  },
  "validation": {
    "rows": 164,
    "features": {
      "correct_answers": "Value('int64')",
      "number_correct_answers": "ClassLabel(names=['1', '2', '3', '4', '5'])"
    },
    "number_correct_answers": {
      "0": 164
    },
    "correct_answers": {
      "4": 24,
      "1": 28,
      "3": 43,
      "0": 21,
      "2": 48
    }
  },
  "test": {
    "rows": 321,
    "features": {
      "correct_answers": "Value('int64')",
      "number_correct_answers": "ClassLabel(names=['1', '2', '3', '4', '5'])"
    },
    "number_correct_answers": {
      "0": 321
    },
    "correct_answers": {
      "2": 85,
      "1": 73,
      "4": 43,
      "3": 72,
      "0": 48
    }
  }
}
```

## medquad

```json
{
  "rows": 16407,
  "qtype": {
    "information": 4535,
    "symptoms": 2748,
    "treatment": 2442,
    "inheritance": 1446,
    "frequency": 1120,
    "genetic changes": 1087,
    "causes": 727,
    "exams and tests": 653,
    "research": 395,
    "outlook": 361,
    "susceptibility": 324,
    "considerations": 235,
    "prevention": 210,
    "stages": 77,
    "complications": 46,
    "support groups": 1
  },
  "duplicate_questions": 2109,
  "duplicate_qa_pairs": 48,
  "empty_answers": 0,
  "tokens_question": {
    "p50": 12,
    "p90": 17,
    "p95": 20,
    "p99": 25,
    "max": 42,
    "nb > 512": 0,
    "nb > 1024": 0,
    "nb > 2048": 0
  },
  "tokens_answer": {
    "p50": 186,
    "p90": 540,
    "p95": 754,
    "p99": 1640,
    "max": 5704,
    "nb > 512": 1838,
    "nb > 1024": 419,
    "nb > 2048": 113
  }
}
```

## ultramedical_preference

```json
{
  "train": {
    "rows": 109353,
    "label_type": {
      "hard": 45765,
      "length": 41130,
      "easy": 22458
    },
    "source": {
      "WikiInstruct": 27493,
      "MedMCQA": 17554,
      "TextBookQA": 16014,
      "ChatDoctor": 11309,
      "MedQA": 10803,
      "MedQA-Evol": 9239,
      "MedQuad": 5855,
      "Medical-Instruct-120k": 4952,
      "PubMedQA": 3293,
      "MedInstruct-52k": 2841
    },
    "verdict_scores_par_label_type": {
      "length": {
        "chosen > rejected": 35077,
        "egalite": 5467,
        "chosen < rejected": 586
      },
      "easy": {
        "chosen > rejected": 22458
      },
      "hard": {
        "chosen > rejected": 45229,
        "chosen < rejected": 536
      }
    }
  },
  "validation": {
    "rows": 2232,
    "label_type": {
      "hard": 925,
      "length": 859,
      "easy": 448
    },
    "source": {
      "WikiInstruct": 557,
      "MedMCQA": 338,
      "TextBookQA": 331,
      "ChatDoctor": 249,
      "MedQA": 225,
      "MedQA-Evol": 207,
      "MedQuad": 106,
      "Medical-Instruct-120k": 99,
      "PubMedQA": 66,
      "MedInstruct-52k": 54
    },
    "verdict_scores_par_label_type": {
      "hard": {
        "chosen > rejected": 915,
        "chosen < rejected": 10
      },
      "length": {
        "chosen > rejected": 724,
        "egalite": 126,
        "chosen < rejected": 9
      },
      "easy": {
        "chosen > rejected": 448
      }
    }
  },
  "test": {
    "rows": 777,
    "label_type": {
      "easy": 238,
      "hard": 196,
      "length": 180,
      "human": 163
    },
    "source": {
      "ChatDoctor": 93,
      "MedQuad": 88,
      "Medical-Instruct-120k": 87,
      "WikiInstruct": 86,
      "MedInstruct-52k": 81,
      "MedQA-Evol": 76,
      "MedMCQA": 75,
      "MedQA": 74,
      "TextBookQA": 73,
      "PubMedQA": 44
    },
    "verdict_scores_par_label_type": {
      "human": {
        "chosen < rejected": 19,
        "egalite": 23,
        "chosen > rejected": 121
      },
      "hard": {
        "egalite": 98,
        "chosen > rejected": 83,
        "chosen < rejected": 15
      },
      "easy": {
        "chosen > rejected": 228,
        "chosen < rejected": 10
      },
      "length": {
        "chosen > rejected": 137,
        "chosen < rejected": 43
      }
    }
  },
  "train_tokens_echantillon": {
    "taille": 5000,
    "prompt": {
      "p50": 84,
      "p90": 199,
      "p95": 276,
      "p99": 450,
      "max": 787,
      "nb > 512": 23,
      "nb > 1024": 0,
      "nb > 2048": 0
    },
    "chosen": {
      "p50": 406,
      "p90": 791,
      "p95": 857,
      "p99": 982,
      "max": 1544,
      "nb > 512": 1900,
      "nb > 1024": 30,
      "nb > 2048": 0
    },
    "rejected": {
      "p50": 331,
      "p90": 767,
      "p95": 842,
      "p99": 988,
      "max": 5502,
      "nb > 512": 1631,
      "nb > 1024": 34,
      "nb > 2048": 1
    }
  }
}
```

## recouvrement_mcq

```json
{
  "doublons_internes": {
    "mediqal/mcqu/train": 291,
    "mediqal/mcqu/validation": 20,
    "mediqal/mcqu/test": 98,
    "mediqal/mcqm/train": 131,
    "mediqal/mcqm/validation": 3,
    "mediqal/mcqm/test": 67,
    "frenchmedmcqa/train": 3,
    "frenchmedmcqa/validation": 0,
    "frenchmedmcqa/test": 0
  },
  "recouvrements": {
    "mediqal/mcqu/train & mediqal/mcqu/validation": 143,
    "mediqal/mcqu/train & mediqal/mcqu/test": 46,
    "mediqal/mcqu/train & mediqal/mcqm/train": 9,
    "mediqal/mcqu/train & mediqal/mcqm/validation": 5,
    "mediqal/mcqu/train & mediqal/mcqm/test": 5,
    "mediqal/mcqu/validation & mediqal/mcqu/test": 10,
    "mediqal/mcqu/validation & frenchmedmcqa/train": 1,
    "mediqal/mcqu/test & mediqal/mcqm/train": 6,
    "mediqal/mcqu/test & mediqal/mcqm/validation": 2,
    "mediqal/mcqu/test & mediqal/mcqm/test": 2,
    "mediqal/mcqu/test & frenchmedmcqa/test": 1,
    "mediqal/mcqm/train & mediqal/mcqm/validation": 60,
    "mediqal/mcqm/train & mediqal/mcqm/test": 39,
    "mediqal/mcqm/validation & mediqal/mcqm/test": 12,
    "frenchmedmcqa/train & frenchmedmcqa/validation": 1,
    "frenchmedmcqa/train & frenchmedmcqa/test": 1,
    "frenchmedmcqa/validation & frenchmedmcqa/test": 1
  }
}
```