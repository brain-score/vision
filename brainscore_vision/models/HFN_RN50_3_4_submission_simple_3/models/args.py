import argparse
import sys
import yaml
import os

args = None

def parse_arguments(config_file):
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet50", help="model architecture"
    )
    parser.add_argument(
        "-n", "--name", metavar="NAME", default="ResNet50", help="model name"
    )
    parser.add_argument(
        # "--config", help="Config file to use (see configs dir)", default=os.path.join(os.getcwd(), config_file)
        "--config", help="Config file to use (see configs dir)", default=config_file
    )
    parser.add_argument("--num-classes", default=1000, type=int)
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--conv-type", type=str, default="DenseConv", help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default="NonAffineBatchNorm", help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--prune-rate",
        default=1.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--score-init-constant",
        type=float,
        default=None,
        help="Sample Baseline Subnet Init",
    )
    
    #BuilderZero
    parser.add_argument(
        "--nonlinearity-2", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument(
        "--init-2", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--scale-fan-2", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--prune-rate-2",
        default=1.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument("--mode-2", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--conv-type-2", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument("--bn-type-2", default=None, help="BatchNorm type")
    parser.add_argument(
        "--freeze-weights-2",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )

    args = parser.parse_args()

    get_config(args)

    return args


def get_config(args):
    # load yaml file
    yaml_txt = open(args.config).read()
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    args.__dict__.update(loaded_yaml)


def run_args(config_file):
    global args
    if args is None:
        args = parse_arguments(config_file)


run_args("config.yaml")
