import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcrf import CRF
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer)


class DebertaCRF(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return {'predictions': predictions}

class NERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id, is_test=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.is_test = is_test
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['Sentence']
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'word_ids': encoding.word_ids()
        }
        return item

test_path = 'data/test.csv'
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test file {test_path} not found")

test_df = pd.read_csv(test_path)
test_df['Sentence'] = test_df['Sentence'].apply(ast.literal_eval)

model_path = os.path.abspath('model')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory {model_path} not found")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DebertaCRF.from_pretrained(model_path)
model.eval()

label2id = model.config.label2id
id2label = model.config.id2label
test_dataset = NERDataset(test_df, tokenizer, label2id, is_test=True)

trainer = Trainer(model=model)
predictions, _, _ = trainer.predict(test_dataset)

pred_tags = []
for i, pred in enumerate(predictions):
    sentence = test_df.iloc[i]['Sentence']
    word_ids = test_dataset[i]['word_ids']
    # Handle both dictionary and NumPy array outputs, as in notebook
    pred_ids = pred['predictions'] if isinstance(pred, dict) else np.argmax(pred, axis=-1)
    tags = []
    current_word_id = None
    token_idx = 0
    for j, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != current_word_id:
            if isinstance(pred_ids, list):
                if token_idx < len(pred_ids):
                    tags.append(id2label[pred_ids[token_idx]])
                    token_idx += 1
                else:
                    tags.append('O')
            else:
                tags.append(id2label[pred_ids[j]])
            current_word_id = word_id
    if len(tags) < len(sentence):
        tags.extend(['O'] * (len(sentence) - len(tags)))
    elif len(tags) > len(sentence):
        tags = tags[:len(sentence)]
    pred_tags.append(tags)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'NER Tag': [str(tags) for tags in pred_tags]
})
submission_df.to_csv('submission.csv', index=False)
print(f"Submission file saved to submission.csv")