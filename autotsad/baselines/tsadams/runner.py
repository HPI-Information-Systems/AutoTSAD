import subprocess
import sys
from pathlib import Path
from venv import EnvBuilder

from .util import ENTITY_MAPPING


def prepare_tsadams_venv(env_dir: Path = Path(".venv-tsadams"), use_existing: bool = True) -> Path:
    if not env_dir.name.startswith(".venv"):
        raise ValueError(f"env_dir must start with '.venv' but is {env_dir}!")
    if not env_dir.exists():
        use_existing=False

    print(f"Creating virtual environment for tsadams in {env_dir}")
    builder = EnvBuilder(system_site_packages=False, clear=not use_existing, symlinks=True, with_pip=True)
    context = builder.ensure_directories(env_dir)
    builder.create(env_dir)

    executable = Path.cwd().resolve() / context.env_exe
    print(f"Using {executable=}")

    if not use_existing:
        print("Installing tsadams in virtual environment")
        cmd = [str(executable), "-m", "pip", "install", "tsadams/src"]
        print(" ".join(cmd))
        code = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=".")
        if code != 0:
            raise ValueError(f"Installing tsadams failed with code {code}!")
    return executable


def run(exec: Path, dataset_hash: str, logfile: Path) -> None:
    entity = ENTITY_MAPPING[dataset_hash]
    command = [
        str(exec), "tsadams/src/scripts/autotsad_main.py",
        "--entity", entity,
        "-c", "autotsad/baselines/tsadams/config.yml"
    ]
    print(f"Executing tsadams on dataset {dataset_hash} ({entity})")
    print(" ".join(command))
    with logfile.open("w") as fh:
        code = subprocess.call(command, stdout=fh, stderr=fh, cwd=".")
    if code != 0:
        raise ValueError(f"Executing tsadams failed with code {code}!")


def main(dataset_hash: str, logfile: Path, use_existing_env: bool = True) -> None:
    env_dir = Path(".venv-tsadams")
    exec = prepare_tsadams_venv(env_dir, use_existing=use_existing_env)
    run(exec, dataset_hash, logfile)


if __name__ == '__main__':
    logfile = Path("./results/tsadams-e02953c5ded77623a4eac7993cda2c7e-mim/execution.log")
    logfile.parent.mkdir(exist_ok=True)
    main("e02953c5ded77623a4eac7993cda2c7e", logfile)
