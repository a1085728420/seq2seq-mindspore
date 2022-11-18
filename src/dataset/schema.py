"""Define schema of mindrecord."""

SCHEMA = {
    "src": {"type": "int64", "shape": [-1]},
    "src_padding": {"type": "int64", "shape": [-1]},
    "prev_opt": {"type": "int64", "shape": [-1]},
    "target": {"type": "int64", "shape": [-1]},
    "tgt_padding": {"type": "int64", "shape": [-1]},
}

TEST_SCHEMA = {
    "src": {"type": "int64", "shape": [-1]},
    "src_padding": {"type": "int64", "shape": [-1]},
}
