"""Base class of data loader."""
import os
import numpy as np

from mindspore.mindrecord import FileWriter
from .schema import SCHEMA, TEST_SCHEMA


class DataLoader:
    """Data loader for dataset."""
    _SCHEMA = SCHEMA
    _TEST_SCHEMA = TEST_SCHEMA

    def __init__(self):
        self._examples = []

    def _load(self):
        raise NotImplementedError

    def padding(self, sen, padding_idx, need_sentence_len=None, dtype=np.int64):
        """Padding <pad> to sentence."""
        if need_sentence_len is None:
            return None
        if sen.shape[0] > need_sentence_len:
            return None
        new_sen = np.array([padding_idx] * need_sentence_len, dtype=dtype)
        new_sen[:sen.shape[0]] = sen[:]
        return new_sen

    def write_to_mindrecord(self, path, train_mode, shard_num=1, desc="seq2seq"):
        """
        Write mindrecord file.

        Args:
            path (str): File path.
            shard_num (int): Shard num.
            desc (str): Description.
        """
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        writer = FileWriter(file_name=path, shard_num=shard_num)
        if train_mode:
            writer.add_schema(self._SCHEMA, desc)
        else:
            writer.add_schema(self._TEST_SCHEMA, desc)
        if not self._examples:
            self._load()

        writer.write_raw_data(self._examples)
        writer.commit()
        print(f"| Wrote to {path}.")

    def _add_example(self, example):
        self._examples.append(example)
