import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
            help="config yaml path")
    parser.add_argument("--load", type=str, default="",
            help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
        help="model running mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
        help="enable evaluation mode for testset")
    parser.add_argument("-d", "--debug", action="store_true",
        help="enable debug mode for test")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args
def build_model(cfg):
    pass
def test_model(_print, cfg, model, test_loader):
    pass
def valid_model(_print, cfg, model, valid_loader, valid_criterion):
    pass
def train_loop(_print, cfg, model, train_loader, criterion, valid_loader, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    pass
def main(args, cfg):
    pass
