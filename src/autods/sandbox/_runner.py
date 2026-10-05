"""Child-process entry point for generated feature engineering code.

Run as a script by the executor, never imported by it:

    python _runner.py <code_path> <input_parquet> <output_parquet> <target_column>

Deliberately imports nothing from autods: the child should load as little as
possible, and nothing that touches configuration or API keys.

Exit codes: 0 ok; 1 the generated code raised (traceback on stderr); 3 the
result breaks the contract in a way Parquet cannot carry back to the parent
(message on stderr after CONTRACT_PREFIX); 4 writing the output failed.
"""

import sys
import traceback

import pandas as pd

ENTRYPOINT = "engineer_features"
CONTRACT_PREFIX = "CONTRACT: "


def main(code_path: str, in_path: str, out_path: str, target_column: str) -> int:
    with open(code_path, encoding="utf-8") as f:
        code = f.read()
    df = pd.read_parquet(in_path)

    namespace: dict = {"__name__": "generated_fe"}
    try:
        # The filename makes traceback line numbers point into the generated code.
        exec(compile(code, "generated_fe.py", "exec"), namespace)
        result = namespace[ENTRYPOINT](df, target_column)
    except Exception:
        traceback.print_exc()
        return 1

    # Parquet cannot represent these, so they would surface as an opaque write
    # error. Report them in contract terms the codegen LLM can act on.
    if not isinstance(result, pd.DataFrame):
        message = f"{ENTRYPOINT} must return a pandas DataFrame, got {type(result).__name__}."
        print(CONTRACT_PREFIX + message, file=sys.stderr)
        return 3
    non_string = [repr(c) for c in result.columns if not isinstance(c, str)]
    if non_string:
        print(
            CONTRACT_PREFIX + f"Column names must be strings; got {non_string[:10]}.",
            file=sys.stderr,
        )
        return 3
    if result.columns.duplicated().any():
        dupes = sorted(set(result.columns[result.columns.duplicated()]))
        print(
            CONTRACT_PREFIX + f"Column names must be unique; duplicated: {dupes[:10]}.",
            file=sys.stderr,
        )
        return 3

    try:
        result.to_parquet(out_path)
    except Exception:
        traceback.print_exc()
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
