#!/usr/bin/env python3

import toml
import json
import os
import argparse
import itertools

#Ole var her
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="""
    Generates json files with all the combinations of
    hyperparameter values from a configuratio file.
    """)

    parser.add_argument('outdir', help='output directory')
    parser.add_argument('config', help='configuration file')

    return parser.parse_args()


def parse_config(path):
    """Parse toml configuration file."""
    with open(path, "r") as config_file:
        return toml.load(config_file)


def main():

    args = parse_args()
    config = parse_config(args.config)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    name = os.path.splitext(os.path.basename(args.config))[0]
    hyperparams = []

    for key, value in config.items():
        hyperparams.append([(key, value) for value in value])

    for num, params in enumerate(itertools.product(*hyperparams)):
        basename = "".join([name, '_', str(num), '.json'])
        with open(os.path.join(args.outdir, basename), "w") as f:
            json.dump(dict(params), f)

if __name__ == '__main__':
    main()
