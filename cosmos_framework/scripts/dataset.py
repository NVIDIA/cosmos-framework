# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate or convert Cosmos conversation and task-aware reasoning datasets."""

import argparse


def main(argv=None):
    from cosmos_framework.data.reasoner.formats import (
        metropolis_to_conversations,  # noqa: F401
        metropolis_to_reasoning,  # noqa: F401
        validate_conversation,  # noqa: F401
        validate_metropolis,  # noqa: F401
        validate_reasoning,  # noqa: F401
    )
    from cosmos_framework.data.reasoner.formats.conversion import BaseConverter
    from cosmos_framework.data.reasoner.formats.validation import BaseValidator

    parser = argparse.ArgumentParser(description=__doc__)
    verbs = parser.add_subparsers(dest="verb", required=True)
    validate = verbs.add_parser("validate").add_subparsers(dest="format", required=True)
    for validator in BaseValidator.formats:
        validator.register_subparser(validate)
    sources = verbs.add_parser("convert").add_subparsers(dest="source", required=True)
    for source in sorted({pair.source_format for pair in BaseConverter.converters}):
        targets = sources.add_parser(source).add_subparsers(dest="target", required=True)
        for converter in BaseConverter.converters:
            if converter.source_format == source:
                converter.register_subparser(targets)
    args = parser.parse_args(argv)
    if args.verb == "validate":
        implementation = next(cls for cls in BaseValidator.formats if cls.format == args.format)
    else:
        implementation = next(
            cls
            for cls in BaseConverter.converters
            if (cls.source_format, cls.target_format) == (args.source, args.target)
        )
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            parser.error("Choose an empty output directory; existing datasets are never overwritten")
    return implementation(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
