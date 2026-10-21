# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Whole-bucket validation and lifetime of borrowed destination references."""

import weakref

import pytest
import torch

from cosmos_framework.posttrain.comm.weight_validation import check_weight_names, copy_named_tensors

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("invalid", ["name", "shape", "dtype"])
def test_later_invalid_entry_leaves_every_destination_untouched(invalid):
    destinations = {"a": torch.zeros(2), "b": torch.zeros(3)}
    before = {name: value.clone() for name, value in destinations.items()}
    later = ("b", torch.ones(3))
    if invalid == "name":
        later = ("unknown", torch.ones(3))
    elif invalid == "shape":
        later = ("b", torch.ones(4))
    else:
        later = ("b", torch.ones(3, dtype=torch.float64))
    with pytest.raises((KeyError, ValueError)):
        copy_named_tensors([("a", torch.ones(2)), later], destinations)
    assert all(torch.equal(value, before[name]) for name, value in destinations.items())


def test_success_copies_strided_sources_and_reports_loaded_names():
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    destination = torch.zeros_like(source, memory_format=torch.contiguous_format)
    assert copy_named_tensors([("a", source)], {"a": destination}) == {"a"}
    assert torch.equal(source, destination)
    assert copy_named_tensors([], {"a": destination}) == set()


def test_retained_validation_exception_does_not_keep_destination_views_alive():
    owners = {"a": torch.zeros(2), "b": torch.zeros(3)}
    views = {name: owner.detach() for name, owner in owners.items()}
    refs = [weakref.ref(view) for view in views.values()]
    with pytest.raises(ValueError) as error:
        copy_named_tensors([("a", torch.ones(2)), ("b", torch.ones(4))], views)
    del views
    assert error.value.__traceback__ is not None
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("actual", [{"a"}, {"a", "b", "extra"}])
def test_name_coverage_rejects_missing_and_extra_destinations(actual):
    with pytest.raises(ValueError, match="coverage.*missing=.*unexpected="):
        check_weight_names({"a", "b"}, actual, context="coverage")
