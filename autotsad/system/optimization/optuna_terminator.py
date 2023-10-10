from optuna.terminator import Terminator, StaticErrorEvaluator, TerminatorCallback

optuna_terminator = Terminator(error_evaluator=StaticErrorEvaluator(0.02), min_n_trials=100)
optuna_terminator_callback = TerminatorCallback(optuna_terminator)
