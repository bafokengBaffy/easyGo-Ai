import argparse

def main():
    parser = argparse.ArgumentParser(description='backfill_features')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: backfill_features.py')
    else:
        print('Executing backfill_features.py')

if __name__ == '__main__':
    main()
