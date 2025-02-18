from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class MLMBiberCollator(DataCollatorForLanguageModeling):
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


@dataclass
class MLMBiberPairCollator(DataCollatorForLanguageModeling):
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 350
    evaluate: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        query_sents = self._encode_text(features, "query_text")
        candidate_sents = self._encode_text(features, "candidate_text")

        query_batch = self._prepare_batch(query_sents)
        candidate_batch = self._prepare_batch(candidate_sents)

        query_encodings = [feature['query_biberPlus'] for feature in features]
        candidate_encodings = [feature['candidate_biberPlus'] for feature in features]

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = query_batch.pop("special_tokens_mask", None)
        special_tokens_mask = candidate_batch.pop("special_tokens_mask", None)

        query_batch["input_ids"], query_batch["labels"] = self.torch_mask_tokens(
            query_batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        candidate_batch["input_ids"], candidate_batch["labels"] = self.torch_mask_tokens(
            candidate_batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        if self.evaluate:
            query_authors = [feature['query_authorID'] for feature in features]
            target_authors = [feature['candidate_authorID'] for feature in features]

            return query_batch, candidate_batch, query_encodings, candidate_encodings, query_authors, target_authors
        else:
            return query_batch, candidate_batch, query_encodings, candidate_encodings

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
