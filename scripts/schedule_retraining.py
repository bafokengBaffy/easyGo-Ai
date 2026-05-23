import argparse

def main():
    parser = argparse.ArgumentParser(description='schedule_retraining')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: schedule_retraining.py')
    else:
        print('Executing schedule_retraining.py')

if __name__ == '__main__':
    main()
