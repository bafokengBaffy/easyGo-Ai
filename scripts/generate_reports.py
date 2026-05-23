import argparse

def main():
    parser = argparse.ArgumentParser(description='generate_reports')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: generate_reports.py')
    else:
        print('Executing generate_reports.py')

if __name__ == '__main__':
    main()
