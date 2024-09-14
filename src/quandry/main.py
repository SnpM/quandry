import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='quandry',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('prompts_path')