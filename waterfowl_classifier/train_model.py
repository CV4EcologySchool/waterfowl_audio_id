import argparse
import yaml
import wandb

import pandas as pd

from opensoundscape import CNN, SpectrogramPreprocessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        required=True,
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config_file, 'r'))
    device = cfg['device']
    number_of_epochs = cfg['number_of_epochs']
    train_df = pd.read_csv(cfg['path_to_train_df'])
    validation_df = pd.read_csv(cfg['path_to_validation_df'])
    architecture = pd.read_csv(cfg["architecture"])
    class_list = pd.read_csv(cfg["class_list"])
    overlay_df = pd.read_csv(cfg["overlay_df"])
    sample_duration = pd.read_csv(cfg["sample_duration"])
    bandpass_minf = pd.read_csv(cfg["bandpass_minf"])
    bandpass_max_f = pd.read_csv(cfg["bandpass_max_f"])
    num_workers = pd.read_csv(cfg["num_workers"])
    batch_size = pd.read_csv(cfg["batch_size"])
    save_path = pd.read_csv(cfg["save_path"])
    wandb_project = pd.read_csv(cfg["wandb_project"])

    wandb.login()

    wandb_session = wandb.init(
        project=wandb_project
    )

    pre = SpectrogramPreprocessor(sample_duration=sample_duration, overlay_df=overlay_df)
    model = CNN(architecture=architecture, sample_duration=sample_duration, classes=class_list)
    model.preprocessor = pre
    model.preprocessor.pipeline.bandpass.set(min_f=bandpass_minf, max_f=bandpass_max_f)
    model.train(train_df, validation_df, epochs=number_of_epochs, num_workers=num_workers, batch_size=batch_size, save_path=save_path, wandb_session=wandb_session)