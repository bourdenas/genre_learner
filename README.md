# Espy Genre Classifier

A genre classifier for espy using a hybrid model that combines:

1. **Dense Network**: Processes categorical features (genres/tags from IGDB, Steam, GOG, Wikipedia)
2. **LSTM Network**: Processes game description text
3. **Feature Fusion**: Combines both feature types for genre predictions


## Architecture Overview

```
Input Layer 1: Dense Features
    ↓
Dense Branch: 2048 → 1024 → 512 units
    ↓
Input Layer 2: Text Sequences (game descriptions)
    ↓
Text Branch: Embedding → Bi-LSTM → Bi-LSTM → Dense(256) → Dense(128)
    ↓
Feature Fusion: Concatenate dense + text features
    ↓
Final Layers: Dense(512) → Dense(256) → Output(50 genres)
```


## Files

- `classifier/models/hybrid.py`: Hybrid model definition
- `train_hybrid.py`: Training script for hybrid model
- `eval_hybrid.py`: Evaluation script for hybrid model


## Key Features

- **External Annotations**: Dense network ingesting anntations from IGDB, Steam, GOG, Wikipedia
- **Text Processing**: LSTM branch processes game descriptions
- **GloVe Support**: Optional pretrained embeddings for better text understanding
- **Bidirectional LSTM**: Captures context from both directions
- **Feature Fusion**: Intelligent combination of dense and text features


## Usage

### 1. Basic Training (Random Embeddings)

```bash
python train_hybrid.py \
    --examples examples_train.csv \
    --model hybrid_model \
    --epochs 50
```

### 2. Training with GloVe Embeddings

```bash
python train_hybrid.py \
    --examples examples_train.csv \
    --mode hybrid_model \
    --glove <glove_path>/glove.6B.100d.txt \
    --epochs 50 \
    --vocab-size 2000 \
    --max-sequence_length 100 \
    --lstm-units 32
```


## Architecture Details

### Dense Branch
- Input: Categorical features from IGDB, Steam, GOG, Wikipedia
- Layers: 2048 → 1024 → 512 units (ReLU activation)
- Dropout: 0.3 between each layer

### Text Branch (LSTM component)
- Input: Game description text (tokenized sequences)
- Embedding: 64d (random) or 50-300d (GloVe)
- LSTM: Bidirectional layers (32 → 16 units)
- Dense: 256 → 128 units for text feature extraction
- Dropout: 0.3 throughout

### Feature Fusion
- Concatenates dense features (512d) + text features (128d)
- Final classification: 512 → 256 → 50 genres
- Softmax activation


## Training Parameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `--vocab-size` | 2000 | Text vocabulary size | 2000-10000 depending on dataset size |
| `--max-sequence-length` | 100 | Max words per description | 100-1000 for game descriptions |
| `--lstm-units` | 32 | LSTM hidden units | 32-128 depending on dataset size |
| `--embedding-dim` | 64 | Embedding dimension (without GloVe) | 64-200 for random embeddings |
| `--trainable-embeddings` | False | Fine-tune GloVe | True for large datasets |


## Evaluation

### Evaluation on labeled data
```bash
python eval_hybrid.py \
    --model hybrid_model \
    --examples examples_eval.csv \
    --threshold .5
```


## File Outputs

After training, you'll have:
- `hybrid_model.keras`: Trained model
- `hybrid_model_tokenizer.pkl`: Text tokenizer for predictions
