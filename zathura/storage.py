from pathlib import Path

from zathura.schema import ExperimentRecord

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "data" / "experiments.jsonl"


def save_record(record: ExperimentRecord, path=DEFAULT_LOG):
    record = ExperimentRecord.model_validate(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(record.model_dump_json() + "\n")


def load_records(path=DEFAULT_LOG):
    path = Path(path)
    if not path.exists():
        return []
    return [ExperimentRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
