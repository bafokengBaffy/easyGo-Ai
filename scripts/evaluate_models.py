import argparse

def main():
    parser = argparse.ArgumentParser(description='evaluate_models')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: evaluate_models.py')
    else:
        print('Executing evaluate_models.py')

if __name__ == '__main__':
    main()
