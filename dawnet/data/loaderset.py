import multiprocessing
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class FilteredDataset(Dataset):
    """Improved dataset with additional functionalities.

    Additional functionalities:
        - Allow filtering data based on some conditions
        - Cache filtered indices to make it faster
    """

    def __init__(self, dataset, indices):
        """Initialize the object"""
        self._indices = indices
        self._original_dataset = dataset

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._original_dataset[self._indices[idx]]


class FilterMixin:
    """Improved dataloader with filtering functionalities.

    Additional functionalities:
        - Allow filtering data based on some conditions
        - Cache filtered indices to make it faster
    """

    def filter(self, conditions=None, names=None, use_cache=True):
        """Filter the dataset based on some condition

        Note: this implementation treats condition and name as list to reuse the
        iteration loop for multiple classes which is to reduce the amount of time.

        # Args:
            conditions <function or [function]>: filtering function[s]
            names <str or [str]>: unique name[s]
            use_cache <bool>: whether to use caching

        # Returns
            <FilteredDataLoader>: the filtered DataLoader
            or
            <{name: FilteredDataLoader}>: in case multiple condition / name
        """
        if callable(conditions):
            conditions = [conditions]

        if isinstance(names, str):
            names = [names]

        if not hasattr(self, "_filter"):
            self._filter = {}

        encoded_names = [self._filter_get_encoded_name(each) for each in names]
        final_datasets = {}

        # get the classes that are in cache
        for name, encoded_name in zip(names, encoded_names):
            if use_cache:
                if encoded_name in self._filter:
                    indices = self._filter[encoded_name]
                    final_datasets[name] = FilteredDataset(self, indices)
                    continue

                indices = self._filter_get_local_cache(encoded_name)
                if indices is not None:
                    self._filter[encoded_name] = indices
                    final_datasets[name] = FilteredDataset(self, indices)
                    continue

        # get the classes that are not in cache
        to_filter = list(zip(names, encoded_names, conditions))
        to_filter = [_ for _ in to_filter if _[0] not in final_datasets]
        indices = defaultdict(list)
        if to_filter:
            # TODO (start) heavy processing if there are a lot of images
            for idx, each_item in enumerate(self):
                # TODO heavy processing if there are a lot of conditions
                for name, encoded_name, condition in to_filter:
                    if condition(idx, each_item):
                        indices[name].append(idx)
                        break
            # TODO (end) heavy processing

            for name, encoded_name, condition in to_filter:
                self._filter[encoded_name] = indices[name]
                self._filter_store_local_cache(encoded_name)
                final_datasets[name] = FilteredDataset(self, indices[name])

        if len(final_datasets) == 1:
            return list(final_datasets.values())[0]

        return final_datasets

    def _filter_get_encoded_name(self, name):
        """Return encoded name

        # Args
            name <str>: the name

        # Returns
            <str>: the encoded name
        """
        encoded_name = f'{self.__class__.__name__}_{name}_{len(self)}'
        return encoded_name

    def _filter_get_local_cache(self, encoded_name):
        """Get the local cache

        # Args
            encoded_name <str>: the encoded name

        # Returns
            <list>: whether local cache exists. None if local cache does not exist
        """
        location = Path(f"~/.dawnet/cache/dataset/{encoded_name}").expanduser()
        if location.is_file():
            with location.open("rb") as f_in:
                indices = pickle.load(f_in)
            return indices

        return

    def _filter_store_local_cache(self, encoded_name):
        """Store the cache locally

        # Args
            encoded_name <str>: the encoded name
        """
        location = Path(f"~/.dawnet/cache/dataset/{encoded_name}").expanduser()
        location.parent.mkdir(parents=True, exist_ok=True)

        with location.open("wb") as f_out:
            pickle.dump(self._filter[encoded_name], f_out)
