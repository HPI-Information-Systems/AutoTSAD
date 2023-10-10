# AutoTSAD

Time series anomaly detection system that uses best-in-class base anomaly detectors to detect subsequence and point anomalies in time series with differing characteristics.

## Requirements

- python >= 3.8

## Usage

```sh
$ autotsad --help
usage: autotsad [-h] [--version] {completion,run,db,estimate-period} ...

Unsupervised anomaly detection system for univariate time series.

positional arguments:
  {completion,run,db,estimate-period}
    completion          Output shell completion script
    run                 Run AutoTSAD on a given dataset.
    db                  Manage AutoTSAD result database.
    estimate-period     Estimate the period size of a given time series dataset.

optional arguments:
  -h, --help            show this help message and exit
  --version             Show version number of AutoTSAD.
```

### Shell Completion

AutoTSAD comes with shell auto-completion scripts for bash and zsh. To enable them, run the following commands:

- Bash:

  ```bash
  autotsad completion bash > /etc/bash_completion.d/autotsad
  ```

- Zsh:

  ```zsh
  autotsad completion zsh > /usr/local/share/zsh/site-functions/_autotsad
  ```

- Zsh (with Oh-My-Zsh):

  ```zsh
  mkdir ~/.oh-my-zsh/completions
  autotsad completion zsh > ~/.oh-my-zsh/completions/_autotsad
  ```
