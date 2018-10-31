import argparse


def get_environment_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="folders to the parent directory of real samples")
    parser.add_argument("--generated", required=True, help="folders to the parent directory of generated samples")
    parser.add_argument("--dumpDir", default="output/", help="dir to dump latent_tsr representations")
    parser.add_argument("--reuse", action="store_true", help="reuse dumped data")
    return parser.parse_args()
