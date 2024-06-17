import json
import gzip
import sqlite3

from loguru import logger


def load_json(filepath, gz=False, **kwargs):
    if gz:
        open_func = gzip.open
    else:
        open_func = open

    with open_func(filepath, "rt", encoding="utf8") as f:
        return json.load(f, **kwargs)


def dump_json(obj, filepath, **kwargs):
    with open(filepath, "w", encoding="utf8") as fout:
        json.dump(obj, fout, **kwargs)


def load_jsonlines_iter(filepath, gz=False, skip_err=False, **kwargs):
    if gz:
        open_func = gzip.open
    else:
        open_func = open

    num_err = 0
    with open_func(filepath, "rt", encoding="utf8") as f:
        for line in f:
            try:
                ins = json.loads(line, **kwargs)
                yield ins
            except json.decoder.JSONDecodeError as err:
                if skip_err:
                    num_err += 1
                else:
                    raise err
    logger.debug(f"Skip {num_err} lines with JSONDecodeError in {filepath}")


def load_jsonlines(filepath, gz=False, skip_err=False, **kwargs):
    data = []
    for ins in load_jsonlines_iter(filepath, gz=gz, skip_err=skip_err, **kwargs):
        data.append(ins)
    return data


def dump_jsonlines(obj, filepath, **kwargs):
    with open(filepath, "w", encoding="utf8") as f:
        for ins in obj:
            f.write(json.dumps(ins, **kwargs) + "\n")


def append_jsonlines(obj: list, filepath, **kwargs):
    with open(filepath, "a", encoding="utf8") as f:
        for ins in obj:
            f.write(json.dumps(ins, **kwargs) + "\n")


def trainer_save_model_safe(trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def sqlite_to_jsonlines(db_path, output_file, table_name):
    """Converts a specific table from an SQLite3 database to JSON Lines format.

    Examples:
        >>> db_path = 'my_database.db'
        >>> output_file = 'output.jsonl'
        >>> table_name = 'products'  # Replace with the name of the table you want to convert
        >>> sqlite_to_jsonlines(db_path, output_file, table_name)
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM {}".format(table_name))
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    data = []
    for row in rows:
        # row bytes to string
        ins = {}
        for col, val in zip(column_names, row):
            if isinstance(val, bytes):
                val = val.decode("utf8")
            ins[col] = val
        data.append(ins)
    dump_jsonlines(data, output_file)

    conn.close()
