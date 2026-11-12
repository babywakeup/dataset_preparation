"""Compatibility imports; the sole training implementation lives in the root script."""


def main(argv=None):
    from train_flux3_recolor_lora import main as train
    return train(argv)


def parse_args(argv=None):
    from train_flux3_recolor_lora import parse_args as parse
    return parse(argv)
