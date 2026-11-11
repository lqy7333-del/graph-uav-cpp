"""Build a frozen manifest from trusted local initial instance snapshots."""
import argparse
from coverage_env.rl.evaluation.framework_comparison import prepare_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cc-results', help='Optional existing CC results directory, seed42 filenames')
    args = p.parse_args()
    prepare_manifest(args.snapshots, args.output, cc_results=args.cc_results)


if __name__ == '__main__': main()
