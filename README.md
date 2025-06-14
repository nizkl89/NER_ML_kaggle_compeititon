# COMP 4211 Machine Learning Group Project: Named Entity Recognition (NER)

## Overview
This repository contains the code and resources for our COMP 4211 Spring 2025 group project, submitted to a Kaggle Named Entity Recognition (NER) competition. The goal was to predict entity types (e.g., organization, person, location) for words in sentences, using a dataset of 40,000 labeled training sentences and 5,000 test sentences. Our team achieved a weighted F1 score of **0.856870** on the private leaderboard, securing **8th place out of 79 teams**. We developed a DeBERTa-base model with a Conditional Random Field (CRF) layer and synonym-based data augmentation to improve performance, particularly for rare classes.

## Project Highlights
- **Task**: Named Entity Recognition (NER) with 19 entity tags (e.g., B-org, I-per, O).
- **Dataset**: 40,000 training sentences and 5,000 test sentences, evaluated via weighted F1 score.
- **Model**: DeBERTa-base (183M parameters) with a custom CRF layer to enforce valid tag sequences.
- **Data Augmentation**: Used nlpaug with WordNet to augment rare classes (B-art, B-eve, B-nat), adding ~10% more examples.
- **Training**: Fine-tuned with Hugging Face Trainer API, learning rate 1e-5, batch size 32 (via gradient accumulation), and 8 epochs.
- **Results**:
  - Validation macro F1: **0.70**
  - Public leaderboard weighted F1: **~0.858**
  - Private leaderboard weighted F1: **0.856870**
  - Rank: **8th / 79 teams**

## Methodology
- **Data Preprocessing**: Converted string representations of sentences and tags into Python lists using `ast.literal_eval`. Created label mappings for 19 NER tags.
- **Augmentation**: Applied synonym-based augmentation for rare classes, preserving tag alignment.
- **Model Architecture**: Used `microsoft/deberta-base` with a custom CRF layer to model tag sequence dependencies, preventing invalid transitions.
- **Training Setup**: Trained on NVIDIA Tesla P100 GPU for ~1 hour 15 minutes, with hyperparameters optimized for macro F1 to prioritize rare class performance.
- **Experiments**: Tested DistilBERT, BERT-base, and RoBERTa, but DeBERTa with CRF outperformed others due to better handling of rare classes and tag sequences.

## Key Insights
- The CRF layer was critical for reducing invalid tag transitions, boosting macro F1 by ~0.05.
- Targeted augmentation for rare classes improved recall for B-art and B-eve without introducing excessive noise.
- A low learning rate (1e-5) and effective batch size of 32 ensured stable convergence.

## Future Improvements
- Explore ensemble models to further boost performance.
- Experiment with advanced augmentation techniques for rare classes.
- Investigate longer training or alternative transformer models.


## Acknowledgments
We thank the COMP 4211 course staff for providing the Kaggle competition framework and resources. This project was completed as part of the Spring 2025 semester at HKUST.
