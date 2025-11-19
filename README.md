# BTP---Samarth-Rathore

# Temporal Entity Extraction and Cross-Lingual NER

**BTP**  
**Student:** Samarth Rathore (22075073)  
**Department:** Computer Science and Engineering (B.Tech)  
**Note:** To properly see the formatted code download the code files and open through jupyter notebook or collab.

***

## Overview

This project addresses two critical challenges in Natural Language Processing for low-resource Indic languages:

### **Task 1: Temporal Entity Extraction using LLM APIs**
Building a robust LLM-based system for extracting temporal entities (TIMEX3 time expressions and EVENT tags) from English TimeBank data, achieving **F1 Score: 0.86** - competitive with state-of-the-art rule-based systems.

### **Task 2: Cross-Lingual NER for 7 Indic Languages**
Generating high-quality parallel NER-annotated datasets for Assamese, Bengali, Gujarati, Malayalam, Marathi, Tamil, and Telugu through cross-lingual projection from Hindi, followed by multilingual model fine-tuning achieving **aggregate F1: 0.789** across all languages.

[Generated Dataset](https://drive.google.com/drive/folders/1V7EETVly_vDpqS0uPqGAsDlv5VMPtwcb?usp=sharing)

### **Task 3: Hindi Temporal Entity Recognition**
Fine-tuning XLM-RoBERTa on [Hindi TimeBank dataset](https://github.com/pranavgoel25/HindiTimeBank) for EVENT, STATE, and TIMEX extraction, achieving **F1: 0.78** on test set.

***

## Project Structure

```
BTP---Samarth-Rathore/
│
├── Temporal_Entity_extraction_using_LLM_API.ipynb
│   └── Temporal entity extraction (TIMEX3, EVENT) using Gemini API
│       - Baseline, dummy tag, dependency parsing, final optimized approach
│
├── DatasetGeneration_with_NER_Tags.ipynb
│   └── Cross-lingual NER dataset generation pipeline
│       - Word alignment (SimAlign/Awesome-align)
│       - Tag projection to target languages
│       - 3-method evaluation (entity matching, tag consistency, token accuracy)
│
├── Finetuning_of_Model.ipynb
│   └── XLM-RoBERTa fine-tuning on generated NER dataset
│       - Training on 7 Indic languages
│       - Per-language and aggregate evaluation
|       - Hindi temporal entity model training
│
├── Evaluation_of_finetuned_model_on_our_Dataset.ipynb
│
└── README.md
    └── This comprehensive documentation
```

***

## Part 1: Temporal Entity Extraction using LLM APIs

### Problem Statement

Extract temporal entities (TIMEX3 and EVENT tags) from English text using the **TimeML/TempEval-3** benchmark dataset

**Goal**: Match or exceed state-of-the-art rule-based systems using LLM API-based methods.

***

### Methodology: 4 Progressive Approaches

#### **Approach 1: Baseline Direct Extraction**

**Method**: Few-shot prompting with Gemini API

**Results**:
| Metric | Score |
|--------|-------|
| Precision | 0.87 |
| Recall | 0.56 |
| **F1 Score** | **0.67** |

***

#### **Approach 2: Dummy Tag Replacement**

**Method**: Replace all temporal tags with unique placeholders

**Results**:
| Metric | Score |
|--------|-------|
| Precision | 0.59 |
| Recall | 0.59 |
| **F1 Score** | **0.59** |

***

#### **Approach 3: Stanford Dependency Parsing**

**Method**: Enhance prompts with syntactic dependency information

**Results**:
| Metric | Score |
|--------|-------|
| Precision | 0.57 |
| Recall | 0.51 |
| **F1 Score** | **0.54** |

***

#### **Approach 4: Final Optimized Method**

**Research-Inspired Improvements** :

1. **Decomposed Prompting**: Separate TIMEX3 and EVENT examples
2. **Filtered Dependency Context**: Only relevant relations and POS tags
3. **Stricter Evaluation**: Threshold raised from 0.6 to 0.7
4. **Post-Processing Rules**: Remove invalid predictions

**Results**:
| Metric | Score |
|--------|-------|
| **Precision** | **0.94** |
| **Recall** | **0.81** |
| **F1 Score** | **0.86** |

***

### Comparison with State-of-the-Art

| Model/System | Approach | F1 Score | Dataset |
|--------------|----------|----------|---------|
| **Ours (Final)** | **Gemini API + Deps + Prompt Eng.** | **0.86** | **TempEval-3** |
| **PTime** | Rule-based pattern matching | **0.87** | TempEval-3 |
| HeidelTime | Rule-based temporal tagger | 0.81 | TempEval-3 |
| SUTime (Stanford) | Regex pattern-based | 0.76 | TempEval-3 |
| NavyTime | ML + linguistic features | 0.79 | TempEval-3 |
| Vicuna LLM | Decomposed prompting | 0.86 | Clinical |

**Key Achievement**: Our LLM-based approach achieves **0.86 F1**, matching the best pure rule-based system (PTime: 0.87) while requiring **zero manual rule engineering**.

***

## Part 2: Cross-Lingual NER Dataset Generation

### Problem Statement

Generate high-quality NER-annotated parallel corpora for **7 low-resource Indic languages** using Hindi (Samanantar) as the pivot language.

**Target Languages**: Assamese, Bengali, Gujarati, Malayalam, Marathi, Tamil, Telugu  
**Labels**: B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O

***

### Methodology Pipeline

1. **Word Alignment**: SimAlign/Awesome-align for token-level alignment
2. **Tag Projection**: Transfer tags via word alignments
3. **Quality Validation**: 3-method evaluation

***

### Evaluation Results

#### **Method 1: Entity Matching Accuracy**

| Language | Entity Ratio
|----------|--------------|
| Assamese | 0.838 
| Bengali | 0.682 
| Gujarati | 0.829 
| Malayalam | 0.610 
| Marathi | 0.733 
| Tamil | 0.846 
| Telugu | 0.758 
| **Average** | **0.757** |

***

#### **Method 2: Tag Type Consistency**

| Language | Similarity Score 
|----------|------------------|
| Assamese | 0.981 
| Bengali | 0.916 
| Gujarati | 0.947 
| Malayalam | 0.910 
| Marathi | 0.884 
| Tamil | 0.874 
| Telugu | 0.900 
| **Average** | **0.916** |

***

#### **Method 3: Token-Level Alignment Accuracy**

| Language | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| Assamese | 83.8% | 0.834 | 0.838 | **0.836** |
| Bengali | 86.8% | 0.750 | 0.830 | **0.820** |
| Gujarati | 76.5% | 0.746 | 0.765 | **0.754** |
| Malayalam | 84.0% | 0.790 | 0.800 | **0.790** |
| Marathi | 85.0% | 0.800 | 0.790 | **0.790** |
| Tamil | 81.6% | 0.834 | 0.816 | **0.816** |
| Telugu | 90.0% | 0.790 | 0.800 | **0.754** |
| **Average** | **87.5%** | **0.792** | **0.806** | **0.820** |


***

## Part 3: Model Fine-tuning and Evaluation

### Model Architecture

**Base Model**: XLM-RoBERTa (xlm-roberta-base)
**Fine-tuning Setup**:
- Task: Token classification
- Training: 10 epochs, learning rate 2e-5-3e-5, batch size 4-8
- Optimization: AdamW with weight decay 0.01

***

### Results: Cross-Lingual NER Model

#### **Per-Language Performance**

| Language | F1 Score | Precision | Recall | Accuracy
|----------|----------|-----------|--------|----------
| **Assamese (as)** | **0.7900** | 0.7704 | 0.8104 | 96.04%
| **Bengali (bn)** | **0.7810** | 0.7610 | 0.8000 | 97.10%
| **Gujarati (gu)** | **0.7538** | 0.8238 | 0.7218 | 97.38% 
| **Malayalam (ml)** | **0.7857** | 0.7389 | 0.8057 | 97.57%
| **Marathi (mr)** | **0.8200** | 0.8450 | 0.8001 | 98.38%
| **Tamil (ta)** | **0.7610** | 0.7557 | 0.7991 | 96.20% 
| **Telugu (te)** | **0.8091** | 0.8377 | 0.7881 | 97.77%

***

#### **Aggregate Performance Across All Languages**

| Metric | Score |
|--------|-------|
| **F1 Score** | **0.7891 (78.91%)** |
| **Precision** | **0.7964** |
| **Recall** | **0.7790** |
| **Accuracy** | **97.97%** |

***

### Results: Hindi TimeBank Temporal Entity Extraction

**Labels**: B-EVENT, I-EVENT, B-STATE, I-STATE, B-TIMEX, I-TIMEX, O

**Label Distribution**:
- Events (B/I-EVENT):(20.1%)
- States (B/I-STATE):(2.9%)
- Time expressions (B/I-TIMEX):(1.4%)
- Outside (O):(75.5%)

***

#### **Test Set Performance**

| Metric | Score |
|--------|-------|
| **Overall F1** | **0.78** |
| **Precision** | **0.80** |
| **Recall** | **0.76** |
| **Accuracy** | **96.8%** |

#### **Entity-wise Performance**

| Entity Type | Precision | Recall | F1 Score |
|-------------|-----------|--------|----------|
| EVENT | 0.82 | 0.79 | 0.80 |
| STATE | 0.76 | 0.72 | 0.74 |
| TIMEX | 0.81 | 0.75 | 0.78 |

***


***

## Complete Results Summary

### Part 1: Temporal Entity Extraction (English)

| Approach | F1 Score |
|----------|----------|
| Baseline | 0.67 | 
| Dummy Tag | 0.59 |
| Dependency | 0.54 |
| **Final** | **0.86** |

***

### Part 2: Cross-Lingual NER Dataset Quality

| Method | Score |  
|--------|-------|  
| Entity Matching | 75.7% | 
| Tag Consistency | 91.6% | 
| Token F1 | 82.0% |

***

### Part 3: Fine-tuned Models

#### A. Cross-Lingual NER
| Metric | Score |
|--------|-------|
| **Aggregate F1** | **78.91%** |
| **Accuracy** | **97.97%** |

#### B. Hindi Temporal Entities
| Metric | Score |
|--------|-------|
| **F1 Score** | **0.78** |
| **Accuracy** | **96.8%** |

***

## License

Academic use only. Datasets derived from publicly available sources.

***

## Author

**Samarth Rathore**  
Roll Number: 22075073  
Computer Science & Engineering (B.Tech)  

***
