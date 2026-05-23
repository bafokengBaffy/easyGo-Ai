import argparse

def main():
    parser = argparse.ArgumentParser(description='monitor_performance')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: monitor_performance.py')
    else:
        print('Executing monitor_performance.py')

if __name__ == '__main__':
    main()
