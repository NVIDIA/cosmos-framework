# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

torch.compile-safe functools wrappers (specifically lru_cache).
"""

import functools


def lru_cache(maxsize=128, typed=False):
    """
    A torch.compile-safe wrapper around functools.lru_cache.

    This decorator automatically disables caching when inside a torch-compiled region.
    torch.compile ignores lru_cache and raises warnings; since torch.compile acts as a
    higher-level cache itself, lru_cache becomes redundant and we disable it to avoid warnings.

    When not in a torch-compiled region, behaves exactly like functools.lru_cache.
    When in a torch-compiled region it's a no-op.
    """

    def decorator(func):
        # Create the cached version using lru_cache
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we're in a torch-compiled region
            from cosmos_framework.model.attention.utils.environment import is_torch_compiling

            if is_torch_compiling():
                # Bypass cache during compilation
                return func(*args, **kwargs)
            else:
                # Use cached version normally
                return cached_func(*args, **kwargs)

        # Expose cache_clear and cache_info methods when not compiling
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        wrapper.__wrapped__ = func

        return wrapper

    # Support both @lru_cache and @lru_cache() syntax
    # If called without parentheses (maxsize is actually the function)
    if callable(maxsize):
        func = maxsize
        maxsize = 128
        return decorator(func)

    return decorator
