from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class MLMTextCollator(DataCollatorForLanguageModeling):
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 350

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        text_sents = self._encode_text(features, "text")
        batch = self._prepare_batch(text_sents)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch

    def _encode_text(self, features, feature_name):
        return [{'input_ids': self.tokenizer(feature[feature_name])['input_ids'][:self.max_length]} for feature in
                features]

    def _prepare_batch(self, sents):
        return self.tokenizer.pad(
            sents,
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )


@dataclass
class MLMTextBiberCollator(DataCollatorForLanguageModeling):
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 350

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        text_sents = self._encode_text(features, "text")
        batch = self._prepare_batch(text_sents)

        encodings = [feature['text_biberPlus'] for feature in features]

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch, encodings

    def _encode_text(self, features, feature_name):
        return [{'input_ids': self.tokenizer(feature[feature_name])['input_ids'][:self.max_length]} for feature in
                features]

    def _prepare_batch(self, sents):
        return self.tokenizer.pad(
            sents,
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
