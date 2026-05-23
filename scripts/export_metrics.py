import argparse

def main():
    parser = argparse.ArgumentParser(description='export_metrics')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: export_metrics.py')
    else:
        print('Executing export_metrics.py')

if __name__ == '__main__':
    main()
