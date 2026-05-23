import argparse

def main():
    parser = argparse.ArgumentParser(description='run_pipeline')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Dry run: run_pipeline.py')
    else:
        print('Executing run_pipeline.py')

if __name__ == '__main__':
    main()
