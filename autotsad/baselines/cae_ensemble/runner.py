import subprocess
import sys
from pathlib import Path
from venv import EnvBuilder


def prepare_cae_ensemble_env(env_dir: Path = Path(".venv-cae-ensemble"), use_existing: bool = True) -> Path:
    if not env_dir.name.startswith(".venv"):
        raise ValueError(f"env_dir must start with '.venv' but is {env_dir}!")
    if not env_dir.exists():
        use_existing = False

    print(f"Creating virtual environment for cae-ensemble in {env_dir}")
    builder = EnvBuilder(system_site_packages=False, clear=not use_existing, symlinks=True, with_pip=True)
    context = builder.ensure_directories(env_dir)
    builder.create(env_dir)

    executable = Path.cwd().resolve() / context.env_exe
    print(f"Using {executable=}")

    if not use_existing:
        print("Installing cae-ensemble dependencies in virtual environment")
        cmd = [str(executable), "-m", "pip", "install", "-r", "autotsad/baselines/cae_ensemble/requirements.txt"]
        print(" ".join(cmd))
        code = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=".")
        if code != 0:
            raise ValueError(f"Installing dependencies for cae-ensemble failed with code {code}!")
    return executable


def run(exec: Path, dataset_hash: str, dataset_path: Path, logfile: Path) -> None:
    command = [
        str(exec), "cae_ensemble.py",
        "--dataset", "-1",
        "--dataset-path", str(dataset_path),
        "--validation", "1",
        "--hyperparameter-samples", "10",
        "--early_stopping", "true",
        "--save_output", "true",
        "--save_config", "true",
        "--save_figure", "false",
        "--save_model", "false",
        "--load_model", "false",
        "--load_config", "false",
        "--server_run", "false",
    ]
    print(f"Executing cae-ensemble on dataset {dataset_hash}")
    print(" ".join(command))
    with logfile.open("w") as fh:
        code = subprocess.call(command, stdout=fh, stderr=fh, cwd="cae-ensemble")
    if code != 0:
        raise ValueError(f"Executing cae-ensemble failed with code {code}!")


def main(dataset_hash: str, dataset_path: Path, logfile: Path, use_existing_env: bool = True) -> None:
    env_dir = Path(".venv-cae-ensemble")
    exec = prepare_cae_ensemble_env(env_dir, use_existing=use_existing_env)
    run(exec, dataset_hash, dataset_path, logfile)


if __name__ == '__main__':
    logfile = Path("./results/tsadams-e02953c5ded77623a4eac7993cda2c7e-mim/execution.log")
    logfile.parent.mkdir(exist_ok=True)
    main("e02953c5ded77623a4eac7993cda2c7e", Path("."), logfile)
