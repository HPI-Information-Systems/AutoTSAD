from typing import Optional

from sqlalchemy import select, delete

from .database import Database


def delete_experiment_results(db: Database, experiment_id: Optional[int] = None, name: Optional[str] = None) -> None:
    if experiment_id is None and name is None:
        raise ValueError("Either experiment_id or name must be specified!")

    if experiment_id is None:
        print(f"Resolving experiment information for {name}")
        with db.begin() as conn:
            res = conn.execute(select(db.experiment_table).where(db.experiment_table.c.name == name)).first()
    else:
        print(f"Resolving experiment information for {experiment_id}")
        with db.begin() as conn:
            res = conn.execute(select(db.experiment_table).where(db.experiment_table.c.id == experiment_id)).first()
    if not res:
        raise ValueError(f"Could not find experiment with name {name} and no experiment ID was specified!")

    experiment_id = res[0]
    name = res[1]
    description = res[2]
    date = res[3]

    print(f"\nDeleting experiment with ID {experiment_id}")
    print(f"{name=}")
    print(f"{description=}")
    print(f"{date=}")
    input(f"\nAre your sure you want to delete all results of this experiment? (confirm with ENTER)")

    with db.begin() as conn:
        # ranking
        res = conn.execute(delete(db.ranking_table).where(db.ranking_table.c.experiment_id == experiment_id))
        print(f"Deleted {res.rowcount} entries from table {db.ranking_table.name}")

        # algorithm scoring
        res = conn.execute(
            delete(db.algorithm_scoring_table).where(db.algorithm_scoring_table.c.experiment_id == experiment_id))
        print(f"Deleted {res.rowcount} entries from table {db.algorithm_scoring_table.name}")

        # algorithm execution
        res = conn.execute(
            delete(db.algorithm_execution_table).where(db.algorithm_execution_table.c.experiment_id == experiment_id))
        print(f"Deleted {res.rowcount} entries from table {db.algorithm_execution_table.name}")

        # autotsad execution
        res = conn.execute(
            delete(db.autotsad_execution_table).where(db.autotsad_execution_table.c.experiment_id == experiment_id))
        print(f"Deleted {res.rowcount} entries from table {db.autotsad_execution_table.name}")

        # experiment
        conn.execute(delete(db.experiment_table).where(db.experiment_table.c.id == experiment_id))
        print(f"Deleted all results from experiment {experiment_id}")
