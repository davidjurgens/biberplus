from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer


@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 350
    evaluate: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        query_sents = self._encode_text(features, "query_text")
        candidate_sents = self._encode_text(features, "candidate_text")

        query_batch = self._prepare_batch(query_sents)
        candidate_batch = self._prepare_batch(candidate_sents)

        if self.evaluate:
            query_authors = [feature["query_authorID"] for feature in features]
            target_authors = [feature["candidate_authorID"] for feature in features]

            return query_batch, candidate_batch, query_authors, target_authors
        else:
            return (
                query_batch,
                candidate_batch,
            )

    def _encode_text(self, features, feature_name):
        return [
            {
                "input_ids": self.tokenizer(feature[feature_name])["input_ids"][
                    : self.max_length
                ]
            }
            for feature in features
        ]

    def _prepare_batch(self, sents):
        return self.tokenizer.pad(
            sents,
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )


@dataclass
class ContrastiveBiberCollator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 350
    evaluate: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        query_sents = self._encode_text(features, "query_text")
        candidate_sents = self._encode_text(features, "candidate_text")

        query_batch = self._prepare_batch(query_sents)
        candidate_batch = self._prepare_batch(candidate_sents)

        query_encodings = [feature["query_biberPlus"] for feature in features]
        candidate_encodings = [feature["candidate_biberPlus"] for feature in features]

        if self.evaluate:
            query_authors = [feature["query_authorID"] for feature in features]
            target_authors = [feature["candidate_authorID"] for feature in features]

            return (
                query_batch,
                candidate_batch,
                query_encodings,
                candidate_encodings,
                query_authors,
                target_authors,
            )
        else:
            return query_batch, candidate_batch, query_encodings, candidate_encodings

    def _encode_text(self, features, feature_name):
        return [
            {
                "input_ids": self.tokenizer(feature[feature_name])["input_ids"][
                    : self.max_length
                ]
            }
            for feature in features
        ]

    def _prepare_batch(self, sents):
        return self.tokenizer.pad(
            sents,
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
