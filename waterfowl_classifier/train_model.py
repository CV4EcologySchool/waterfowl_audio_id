import argparse
from datetime import datetime
import os
import shutil
import yaml
import wandb

import pandas as pd

from opensoundscape import CNN, SpectrogramPreprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        required=True,
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config_file, "r"))
    number_of_epochs = cfg["number_of_epochs"]
    train_df = pd.read_csv(cfg["path_to_train_df"], index_col=[0, 1, 2])
    validation_df = pd.read_csv(cfg["path_to_validation_df"], index_col=[0, 1, 2])
    architecture = cfg["architecture"]
    class_list = cfg["class_list"]
    overlay_df = pd.read_csv(cfg["overlay_df"], index_col=[0, 1, 2])
    sample_duration = cfg["sample_duration"]
    bandpass_minf = cfg["bandpass_minf"]
    bandpass_max_f = cfg["bandpass_max_f"]
    num_workers = cfg["num_workers"]
    batch_size = cfg["batch_size"]
    experiment_dir_path = cfg["experiment_dir_path"]
    wandb_project = cfg["wandb_project"]

    todays_date = datetime.now().strftime("%Y%m%d%H%M%S")
    wandb.login()

    wandb_session = wandb.init(project=wandb_project)
    save_path = f"{experiment_dir_path}/experiment_{todays_date}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save config to experiment folder
    shutil.copyfile(args.config_file, f"{save_path}/config.yaml")

    pre = SpectrogramPreprocessor(
        sample_duration=sample_duration,
        overlay_df=overlay_df,
    )
    model = CNN(
        architecture=architecture, sample_duration=sample_duration, classes=class_list
    )
    model.preprocessor = pre
    model.preprocessor.pipeline.bandpass.set(min_f=bandpass_minf, max_f=bandpass_max_f)
    model.preprocessor.pipeline.overlay.set(update_labels=True)
    model.train(
        train_df,
        validation_df,
        epochs=number_of_epochs,
        num_workers=num_workers,
        batch_size=batch_size,
        save_path=save_path,
        wandb_session=wandb_session,
    )

    wandb.unwatch(model.network)
    wandb.finish()
