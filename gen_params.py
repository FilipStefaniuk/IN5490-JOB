import toml
import json
import os
import argparse
import itertools


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="""
        Generates json files with all the combinations of
        hyperparameter values from a configuratio file.
    """)

    parser.add_argument('config', type=argparse.FileType(), help='configuration file')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--basename', default="", help="basename for param files")

    return parser.parse_args()


def main():

    args = parse_args()
    config = toml.load(args.config)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    params = [[(key, value) for value in value] for key, value in config.items()]

    for num, params in enumerate(itertools.product(*params)):
        basename = "".join([args.basename, str(num), '.json'])
        with open(os.path.join(args.outdir, basename), "w") as f:
            json.dump(dict(params), f)

if __name__ == '__main__':
    main()
