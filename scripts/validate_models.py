import argparse

def main():
    parser = argparse.ArgumentParser(description='validate_models')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: validate_models.py')
    else:
        print('Executing validate_models.py')

if __name__ == '__main__':
    main()
