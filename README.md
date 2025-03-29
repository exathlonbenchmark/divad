[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# OmniAnomaly Integration

Running the OmniAnomaly method requires Python 3.6.15, and therefore a different branch and environment.

## Project Configuration

```bash
## OPTION 1: virtualenv
$ virtualenv -p 3.6.15 venv
$ source venv/Scripts/activate venv # Windows with Git Bash or WSL
$ venv/Scripts/activate.bat # Windows with cmd
$ source venv/bin/activate # POSIX systems

## OPTION 2: conda
$ conda create -n exathlon-omni python=3.6.15
$ conda activate exathlon-omni
```

Then install the project and dependencies by running:

```bash
$ pip install -e .[all]
```

Where `[all]` includes:

- `[dev]`: development dependencies (formatting and testing).
- `[docs]`: dependencies for building documentation.
- `[profiling]`: memory profiling dependencies.
- `[notebooks]`: dependencies only used for/in notebooks.

At the root of the project folder, create a `.env` file containing the lines:

```txt
OUTPUTS=path/to/pipeline/outputs-omni
SPARK=path/to/extracted/data/raw
{OTHER_NAME}=path/to/other/data
...
```

## Experiments

Scripts to reproduce our Spark streaming experiments are located under `scripts/spark`.

We ran `Omni` for a latent dimension in [64, 128], using the following scripts:

```bash
## latent=64
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 64 200 200 True True 1e-4 nf 10 3e-5 20 0.5 10.0 -5 40 64 1024 1
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 64 200 200 True True 1e-4 nf 10 1e-4 20 0.5 10.0 -5 40 64 1024 1
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 64 200 200 True True 1e-4 nf 10 3e-4 20 0.5 10.0 -5 40 64 1024 1
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 64 200 200 True True 1e-4 nf 10 1e-3 20 0.5 10.0 -5 40 64 1024 1
# best validation loss for learning rate 3e-4
./evaluate_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 64 200 200 True True 1e-4 nf 10 1e-3 20 0.5 10.0 -5 40 64 1024 1
## latent=128
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 128 200 200 True True 1e-4 nf 10 3e-5 20 0.5 10.0 -5 40 64 1024 1
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 128 200 200 True True 1e-4 nf 10 1e-4 20 0.5 10.0 -5 40 64 1024 1
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 128 200 200 True True 1e-4 nf 10 3e-4 20 0.5 10.0 -5 40 64 1024 1
./train_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 128 200 200 True True 1e-4 nf 10 1e-3 20 0.5 10.0 -5 40 64 1024 1
# best validation loss for learning rate 3e-4
./evaluate_omni_anomaly.sh weakly -1 "1 2 3 4 5 6 9 10" os_only 0 0 0.2 random 7 regular_scaling minmax 20 1 1 1 mean app-settings-rate 128 200 200 True True 1e-4 nf 10 3e-4 20 0.5 10.0 -5 40 64 1024 1
```