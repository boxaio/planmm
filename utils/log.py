import sys
import os
import logging
import glob
from datetime import datetime
import atexit
import wandb
import torch
from pathlib import Path

from configs.env_paths import BASE_DIR

COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'cyan': '\033[96m',
    'normal': '\033[0m',
}

def _colored(msg, color, end_color="cyan"):
    return f"{COLORS.get(color, COLORS['cyan'])}{msg}{COLORS.get(end_color, COLORS['cyan'])}"



class ColorFormatter(logging.Formatter):
    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = _colored("WARNING", "yellow")
        elif record.levelno in ( logging.ERROR, logging.CRITICAL):
            prefix = _colored("ERROR", "red")
        else:
            prefix = _colored("", "cyan")
        return f"{prefix} {log}"


def get_logger(name, level=logging.DEBUG, root=True, log_dir=Path(BASE_DIR)/"logs"):
    """
    Replaces the standard library logging.getLogger call in order to make some configuration
    for all loggers.
    :param name: pass the __name__ variable
    :param level: the desired log level
    :param root: call only once in the program
    :param log_dir: if root is set to True, this defines the directory where a log file is going
                    to be created that contains all logging output
    :return: the logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not root:
        return logger

    # create handler for console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = ColorFormatter(
        _colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False 

    if log_dir is not None:
        log_dir = Path(log_dir)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"{timestamp}.log"

        # open stream and make sure it will be closed
        stream = log_file.open(mode="w")
        atexit.register(stream.close)

        formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                        datefmt="%m/%d %H:%M:%S")
        file_handler = logging.StreamHandler(stream)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_dict(writer, dictionary, prefix="", global_step=None):
    for key, value in dictionary.items():
        full_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(value, dict):
            log_dict(writer, value, full_key, global_step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(full_key, value, global_step)
        else:
            try:
                scalar_value = float(value)
                writer.add_scalar(full_key, scalar_value, global_step)
            except (ValueError, TypeError):
                print(f"Skipping {full_key}: cannot convert to scalar")


class WandbLogger(object):
    def __init__(self, project_name, run_name, output_fodler, id=None):
        os.makedirs(output_fodler)
        self.output_folder = output_fodler
        self.id = id
        self.wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            sync_tensorboard=True,
            dir=output_fodler,
            id=id,
        )

    def add_config(self, config: dict):
        wandb.config.update(config)

    def watch_model(self, model):
        wandb.watch(model)

    def log_values(self, epoch: int, values: dict):
        # wandb.log(vals, step=epoch) # epoch into dict or not?
        vals = values.copy()
        # vals['epoch'] = epoch
        wandb.log(vals, step=epoch)
        # wandb.log(vals)

    def log_3D_shape(self, epoch: int, models: dict):
        shapes = {key: wandb.Object3D(value) for key, value in models.items() }
        wandb.log(shapes, step=epoch)

    def log_image(self, epoch, images: dict):
        ims = {key: wandb.Image(value) for key, value in images.items()}
        wandb.log(ims, step=epoch)

    def get_experiment_id(self):
        return self.id

    def save(self, filename):
        wandb.save(filename)

    def sync(self):
        if 'WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == "dryrun":
            with open(os.path.join(self.output_folder, "wandb", "sync_status.txt"), 'w') as f:
                f.write("not_synced\n")

            # doesn't work on the clusters
            # print("Syncing wandb")
            # cwd = os.getcwd()
            # os.chdir(os.path.join(self.output_folder, "wandb"))
            # os.system('wandb sync')
            # os.chdir(cwd)
            # print("Wandb synced")
        else:
            print("Wandb was synced automatically")


def write_mean_summaries(writer, metrics, abs_step, mode="training", optimizer=None):
    for key in metrics:
        writer.add_scalars(
            main_tag=key, tag_scalar_dict={'%s_Average' % mode: metrics[key]},
            global_step=abs_step, walltime=None,
        )
    if optimizer is not None:
        writer.add_scalar('learn_rate', optimizer.param_groups[0]["lr"], abs_step)


def load_from_checkpoint(net, checkpoint, partial_restore=False, device=None):
    
    assert checkpoint is not None, "no path provided for checkpoint, value is None"

    if os.path.isdir(checkpoint):
        latest_checkpoint = max(glob.iglob(checkpoint + '/*.pth'), key=os.path.getctime)
        print("loading model from %s" % latest_checkpoint)
        saved_net = torch.load(latest_checkpoint)
    elif os.path.isfile(checkpoint):
        print("loading model from %s" % checkpoint)
        if device is None:
            saved_net = torch.load(checkpoint)
        else:
            saved_net = torch.load(checkpoint, map_location=device)
    else:
        raise FileNotFoundError(f"provided checkpoint {checkpoint} not found, does not mach any directory or file.")

    # For partially restoring a model from checkpoint restore only the common parameters and randomly initialize the
    # remaining ones
    if partial_restore:
        net_dict = net.state_dict()
        saved_net = {k: v for k, v in saved_net.items() if (k in net_dict)}
        print("parameters to keep from checkpoint:")
        print(saved_net.keys())
        extra_params = {k: v for k, v in net_dict.items() if k not in saved_net}
        print("parameters to randomly init:")
        print(extra_params.keys())
        for param in extra_params:
            saved_net[param] = net_dict[param]

    net.load_state_dict(saved_net, strict=True)


if __name__ == "__main__":
    logger = get_logger(__name__, root=True)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
