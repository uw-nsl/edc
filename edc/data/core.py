from __future__ import annotations

from typing import TYPE_CHECKING

import multiprocessing as mp
from functools import cached_property
from zipfile import ZipFile

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .. import utils
from .preprocess import get_edc_special_tokens, preprocess_data, make_shared_encoder_data, \
    make_standalone_encoder_data, make_decoder_data

if TYPE_CHECKING:
    from ..types import TODMetadata, TODDialog, Tokenizer

__all__ = [
    "METADATA_PATH",
    "EDCDataModule"
]

METADATA_PATH = "metadata.json"

class EDCDatasetBase:
    def __init__(self, dataset_path: str, subset: str, transformer_name: str, max_rounds: int,
        max_ctx_len: int, slot_name_weight: float, non_copied_value_weight: float,
        with_user_action: bool, with_system_action: bool, one_pass: bool, with_index: bool = False):
        self.dataset_path = dataset_path
        self.subset = subset
        self.transformer_name = transformer_name
        self.max_rounds = max_rounds
        self.max_ctx_len = max_ctx_len
        self.slot_name_weight = slot_name_weight
        self.non_copied_value_weight = non_copied_value_weight
        self.with_user_action = with_user_action
        self.with_system_action = with_system_action
        self.one_pass = one_pass
        self.with_index = with_index
    
    @cached_property
    def _dataset_archive(self) -> ZipFile:
        return ZipFile(self.dataset_path)
    
    @cached_property
    def _tokenizer(self) -> Tokenizer:
        # Load pre-trained tokenizer
        tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
            self.transformer_name, add_prefix_space=True
        )
        # Add special tokens
        tokenizer.add_tokens(get_edc_special_tokens(self.max_rounds), special_tokens=True)

        return tokenizer
    
    def __reduce__(self):
        return type(self), (
            self.dataset_path,
            self.subset,
            self.transformer_name,
            self.max_rounds,
            self.max_ctx_len,
            self.slot_name_weight,
            self.non_copied_value_weight,
            self.with_user_action,
            self.with_system_action,
            self.one_pass,
            self.with_index
        )

class EDCSharedDataset(EDCDatasetBase):
    @cached_property
    def _subset_dialogs(self) -> list[str]:
        # Get dialog paths for subset
        metadata: TODMetadata = utils.load_json(METADATA_PATH, self._dataset_archive)
        return metadata["subsets"][self.subset]
    
    def __len__(self) -> int:
        return len(self._subset_dialogs)
    
    def __getitem__(self, idx: int):
        tokenizer = self._tokenizer

        # Load dialog from archive
        dialog_path = self._subset_dialogs[idx]
        dialog: TODDialog = utils.load_json(dialog_path, self._dataset_archive)
        
        # Pre-process dialog data
        ctx_segments, target_seqs = preprocess_data(
            tokenizer, dialog, self.max_rounds, self.max_ctx_len, self.slot_name_weight,
            self.non_copied_value_weight, self.with_user_action, self.with_system_action,
            self.one_pass
        )
        # Make encoder and decoder data
        encoder_data = make_shared_encoder_data(ctx_segments, tokenizer.bos_token_id)
        decoder_data = make_decoder_data(target_seqs)
        # Make sample
        sample = encoder_data, decoder_data
        if self.with_index:
            sample = dialog_path, sample
        
        return sample

class EDCStandaloneDataset(EDCDatasetBase):
    @cached_property
    def _subset_rounds(self) -> list[tuple[str, int]]:
        # Get dialog paths for subset
        metadata: TODMetadata = utils.load_json(METADATA_PATH, self._dataset_archive)
        dialog_paths = metadata["subsets"][self.subset]

        subset_rounds: list[tuple[str, int]] = []
        # Collect rounds, identified dialog path and round index
        for dialog_path in dialog_paths:
            dialog: TODDialog = utils.load_json(dialog_path, self._dataset_archive)
            for i in range(len(dialog["rounds"])):
                subset_rounds.append((dialog_path, i))
        
        return subset_rounds
    
    def __len__(self) -> int:
        return len(self._subset_rounds)
    
    def __getitem__(self, idx: int):
        tokenizer = self._tokenizer

        # Load dialog from archive
        dialog_path, round_idx = self._subset_rounds[idx]
        dialog: TODDialog = utils.load_json(dialog_path, self._dataset_archive)

        # Pre-process dialog data
        ctx_segments, target_seqs = preprocess_data(
            tokenizer, dialog, self.max_rounds, self.max_ctx_len, self.slot_name_weight,
            self.non_copied_value_weight, self.with_user_action, self.with_system_action,
            self.one_pass, standalone_round_idx=round_idx
        )
        # Make encoder and decoder data
        encoder_data = make_standalone_encoder_data(ctx_segments, tokenizer.bos_token_id)
        decoder_data = make_decoder_data(target_seqs, standalone_ctx=True)
        # Make sample
        sample = encoder_data, decoder_data
        if self.with_index:
            sample = (dialog_path, round_idx), sample

        return sample

class EDCDataModule(LightningDataModule):
    def __init__(self, dataset_path: str, transformer_name: str, max_rounds: int, max_ctx_len: int,
        slot_name_weight: float, non_copied_value_weight: float, with_user_action: bool,
        with_system_action: bool, one_pass: bool, standalone_ctx: bool, n_workers: int,
        predict_subset: str):
        super().__init__()

        self.dataset_path = dataset_path
        self.transformer_name = transformer_name
        self.max_rounds = max_rounds
        self.max_ctx_len = max_ctx_len
        self.slot_name_weight = slot_name_weight
        self.non_copied_value_weight = non_copied_value_weight
        self.with_user_action = with_user_action
        self.with_system_action = with_system_action
        self.one_pass = one_pass
        self.standalone_ctx = standalone_ctx
        self.n_workers = n_workers
        self.predict_subset = predict_subset

        self._mp_ctx = mp.get_context("spawn")
    
    def _make_loader(self, subset: str, shuffle: bool = False, with_index: bool = False
        ) -> DataLoader:
        # Dataset factory
        dataset_factory = EDCStandaloneDataset if self.standalone_ctx else EDCSharedDataset
        # Create dataset
        dataset = dataset_factory(
            self.dataset_path,
            subset,
            self.transformer_name,
            self.max_rounds,
            self.max_ctx_len,
            self.slot_name_weight,
            self.non_copied_value_weight,
            self.with_user_action,
            self.with_system_action,
            self.one_pass,
            with_index
        )

        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.n_workers,
            shuffle=shuffle,
            multiprocessing_context=self._mp_ctx,
            persistent_workers=True
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader("dev")
    
    def predict_dataloader(self) -> DataLoader:
        return self._make_loader(self.predict_subset, with_index=True)
