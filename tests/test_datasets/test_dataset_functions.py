# License: BSD 3-Clause
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
import pandas as pd
import pytest
from oslo_concurrency import lockutils

import openml
from openml import OpenMLDataset
from openml.testing import TestBase


class TestOpenMLDataset(TestBase):
    _multiprocess_can_split_ = True

    def tearDown(self):
        self._remove_pickle_files()
        super().tearDown()

    def _remove_pickle_files(self):
        self.lock_path = os.path.join(openml.config.get_cache_directory(), "locks")  # noqa: PTH118
        for did in ["-1", "2"]:
            with lockutils.external_lock(
                name=f"datasets.functions.get_dataset:{did}",
                lock_path=self.lock_path,
            ):
                pickle_path = os.path.join(  # noqa: PTH118
                    openml.config.get_cache_directory(),
                    "datasets",
                    did,
                    "dataset.pkl.py3",
                )
                try:  # noqa: SIM105
                    os.remove(pickle_path)  # noqa: PTH107
                except (OSError, FileNotFoundError):
                    #  Replaced a bare except. Not sure why either of these would be acceptable.
                    pass

    def _get_empty_param_for_dataset(self):
        return {
            "name": None,
            "description": None,
            "creator": None,
            "contributor": None,
            "collection_date": None,
            "language": None,
            "licence": None,
            "default_target_attribute": None,
            "row_id_attribute": None,
            "ignore_attribute": None,
            "citation": None,
            "attributes": None,
            "data": None,
        }

    def _check_dataset(self, dataset):
        assert type(dataset) == dict  # noqa: E721
        assert len(dataset) >= 2
        assert "did" in dataset
        assert isinstance(dataset["did"], int)
        assert "status" in dataset
        assert isinstance(dataset["status"], str)
        assert dataset["status"] in ["in_preparation", "active", "deactivated"]

    def _check_datasets(self, datasets):
        for did in datasets:
            self._check_dataset(datasets[did])

    @pytest.mark.test_server
    def test_list_datasets_length(self):
        datasets = openml.datasets.list_datasets()
        assert len(datasets) >= 100

    @pytest.mark.test_server
    def test_list_datasets_paginate(self):
        size = 10
        max = 100  # noqa: A001
        for i in range(0, max, size):
            datasets = openml.datasets.list_datasets(offset=i, size=size)
            assert len(datasets) == size
            assert len(datasets.columns) >= 2
            assert "did" in datasets.columns
            assert datasets["did"].dtype == int
            assert "status" in datasets.columns
            assert datasets["status"].dtype == pd.CategoricalDtype(
                categories=["in_preparation", "active", "deactivated"],
            )

    @pytest.mark.test_server
    def test_list_datasets_empty(self):
        datasets = openml.datasets.list_datasets(tag="NoOneWouldUseThisTagAnyway")
        assert datasets.empty


    @pytest.mark.test_server
    def test_get_datasets_by_name(self):
        # did 1 and 2 on the test server:
        dids = ["anneal", "kr-vs-kp"]
        datasets = openml.datasets.get_datasets(dids)
        assert len(datasets) == 2
        _assert_datasets_retrieved_successfully([1, 2])

    @pytest.mark.test_server
    def test_get_datasets(self):
        dids = [1, 2]
        datasets = openml.datasets.get_datasets(dids)
        assert len(datasets) == 2
        _assert_datasets_retrieved_successfully([1, 2])

    @pytest.mark.test_server
    def test_publish_dataset(self):
        arff_file_path = (
            self.static_cache_dir / "org" / "openml" / "test" / "datasets" / "2" / "dataset.arff"
        )
        dataset = OpenMLDataset(
            "anneal",
            "test",
            data_format="arff",
            version=1,
            licence="public",
            default_target_attribute="class",
            data_file=arff_file_path,
        )
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.dataset_id)
        TestBase.logger.info(
            f"collected from {__file__.split('/')[-1]}: {dataset.dataset_id}",
        )
        assert isinstance(dataset.dataset_id, int)

def _assert_datasets_have_id_and_valid_status(datasets: pd.DataFrame):
    assert pd.api.types.is_integer_dtype(datasets["did"])
    assert {"in_preparation", "active", "deactivated"} >= set(datasets["status"])


def _dataset_file_is_downloaded(did: int, file: str):
    cache_directory = Path(openml.config.get_cache_directory()) / "datasets" / str(did)
    return (cache_directory / file).exists()


def _dataset_description_is_downloaded(did: int):
    return _dataset_file_is_downloaded(did, "description.xml")


def _dataset_qualities_is_downloaded(did: int):
    return _dataset_file_is_downloaded(did, "qualities.xml")


def _dataset_features_is_downloaded(did: int):
    return _dataset_file_is_downloaded(did, "features.xml")


def _dataset_data_file_is_downloaded(did: int):
    cache_directory = Path(openml.config.get_cache_directory()) / "datasets" / str(did)
    return any(f.suffix in (".pq", ".arff") for f in cache_directory.iterdir())


def _assert_datasets_retrieved_successfully(
    dids: Iterable[int],
    with_qualities: bool = False,  # noqa: FBT002
    with_features: bool = False,  # noqa: FBT002
    with_data: bool = False,  # noqa: FBT002
):
    """Checks that all files for the given dids have been downloaded.

    This includes:
        - description
        - qualities
        - features
        - absence of data arff if metadata_only, else it must be present too.
    """
    for did in dids:
        assert _dataset_description_is_downloaded(did)

        has_qualities = _dataset_qualities_is_downloaded(did)
        assert has_qualities if with_qualities else not has_qualities

        has_features = _dataset_features_is_downloaded(did)
        assert has_features if with_features else not has_features

        has_data = _dataset_data_file_is_downloaded(did)
        assert has_data if with_data else not has_data