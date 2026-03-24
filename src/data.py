# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: RL_for_pRNN
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Occupancy Heatmaps for wandb data

# %%
# !pwd

# %% [markdown]
# ## Imports and Function Definitions
# %%
import torch
import sys
import types
import re
import pickle
import zipfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from jaxtyping import Float
from typing import Literal
from pathlib import Path
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

DATA_DIR = Path("../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Overall functions

# %%
_NP_DTYPE_MAP = {
    'FloatStorage': np.float32,
    'DoubleStorage': np.float64,
    'LongStorage': np.int64,
    'IntStorage': np.int32,
    'ShortStorage': np.int16,
    'ByteStorage': np.uint8,
    'BoolStorage': np.bool_,
    'HalfStorage': np.float16,
}


def load_eval_trajectories(path: Path) -> dict:
    """Load trajectories.pt, mocking the missing `scripts` module."""
    scripts_pkg = types.ModuleType('scripts')
    scripts_pkg.__path__ = []
    scripts_pkg.__file__ = '/dev/null'

    class AutoModule(types.ModuleType):
        def __getattr__(self, name):
            cls = type(name, (object,), {
                '__init__': lambda self, *a, **kw: None,
                '__setstate__': lambda self, s: self.__dict__.update(s),
            })
            setattr(self, name, cls)
            return cls

    auto_analysis = AutoModule('scripts.analysis_OMT')
    auto_analysis.__file__ = '/dev/null'
    setattr(scripts_pkg, 'analysis_OMT', auto_analysis)

    _prev = {k: sys.modules[k] for k in ('scripts', 'scripts.analysis_OMT') if k in sys.modules}
    sys.modules['scripts'] = scripts_pkg
    sys.modules['scripts.analysis_OMT'] = auto_analysis
    try:
        return torch.load(Path(path), weights_only=False)
    finally:
        for k in ('scripts', 'scripts.analysis_OMT'):
            if k in _prev:
                sys.modules[k] = _prev[k]
            else:
                sys.modules.pop(k, None)


def _parse_pt_file(path: Path) -> tuple[dict[str, str | None], dict[int, int] | None]:
    """
    Read a trajectories.pt without loading binary storage data.

    Parses only the pickle header to extract tensor shapes, then reads only
    the labels storage file (~1.4 MB) rather than the full file (~500 MB).

    Returns:
        shapes: {key: shape_str or None}
        label_counts: {label_value: count} or None if labels key is absent
    """
    tensor_keys = ['obs', 'obs_pred', 'obs_next', 'act', 'hidden_states', 'labels']

    class _FakeStorage:
        def __init__(self, key, dtype):
            self.key = key
            self.dtype = dtype

    class _MetaUnpickler(pickle.Unpickler):
        def __init__(self, f):
            super().__init__(f)
            self._id_to_storage: dict[int, _FakeStorage] = {}

        def persistent_load(self, pid):
            _, storage_cls, key, _location, _numel = pid
            s = _FakeStorage(key, _NP_DTYPE_MAP.get(storage_cls.__name__, np.float32))
            self._id_to_storage[id(s)] = s
            return s

        def find_class(self, module, name):
            if name == '_rebuild_tensor_v2':
                def _capture(storage, offset, size, stride, *args, **kwargs):
                    return {'__sid__': id(storage), '__size__': size, '__offset__': offset}
                return _capture
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return type(name, (object,), {
                    '__init__': lambda self, *a, **kw: None,
                    '__setstate__': lambda self, s: self.__dict__.update(s),
                })

    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split('/')[0]
        with zf.open(f'{prefix}/data.pkl') as f:
            unpickler = _MetaUnpickler(f)
            result = unpickler.load()

        shapes: dict[str, str | None] = {}
        for k in tensor_keys:
            v = result.get(k)
            shapes[k] = str(tuple(v['__size__'])) if isinstance(v, dict) and '__size__' in v else None

        label_counts: dict[int, int] | None = None
        lbl_meta = result.get('labels')
        if isinstance(lbl_meta, dict) and '__sid__' in lbl_meta:
            storage = unpickler._id_to_storage[lbl_meta['__sid__']]
            with zf.open(f'{prefix}/data/{storage.key}') as f:
                raw = np.frombuffer(f.read(), dtype=storage.dtype)
            numel = int(np.prod(lbl_meta['__size__']))
            arr = raw[lbl_meta['__offset__']: lbl_meta['__offset__'] + numel]
            unique, counts = np.unique(arr, return_counts=True)
            label_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    return shapes, label_counts


def get_goal_dataframes(
    goal: str,
    data_dir: Path = DATA_DIR,
    _pool: ProcessPoolExecutor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a given goal string (e.g. '711', '72', '147'), load all matching
    trajectories.pt files and return:
      - shapes_df:  index=tensor key, columns=step count, values=shape tuple
      - labels_df:  index=label value (0/1), columns=step count, values=count

    Pass a shared ``_pool`` to avoid spawning subprocesses on every call.
    """
    tensor_keys = ['obs', 'obs_pred', 'obs_next', 'act', 'hidden_states', 'labels']
    pattern = re.compile(rf"data_cur_lroom_step(\d+)_goal{re.escape(goal)}$")
    dirs = sorted(
        [(int(m.group(1)), d) for d in data_dir.iterdir() if (m := pattern.match(d.name))],
        key=lambda x: x[0],
    )

    paths = [d / 'trajectories.pt' for _, d in dirs]
    executor = _pool or ProcessPoolExecutor()
    try:
        parsed = list(executor.map(_parse_pt_file, paths))
    finally:
        if _pool is None:
            executor.shutdown(wait=False)

    shapes_data = {step: shapes for step, (shapes, _) in zip([s for s, _ in dirs], parsed)}
    labels_data = {step: lc for step, (_, lc) in zip([s for s, _ in dirs], parsed) if lc is not None}

    shapes_df = pd.DataFrame(shapes_data, index=tensor_keys).rename_axis('key')
    labels_df = pd.DataFrame(labels_data).rename_axis('label').sort_index()

    for label in labels_df.index:
        labels_df.loc[f'Class {label} %'] = ((labels_df.loc[label] / labels_df.sum()) * 100).values    
    
    return shapes_df, labels_df


# %%
_pool = ProcessPoolExecutor()

# %% [markdown]
# ## Goal location: [7, 2]
# %%
shapes_72, labels_72 = get_goal_dataframes('72', _pool=_pool)

# %%
if __name__ == "__main__":
    print("=== Goal [7, 2] — Shapes ===")
    print(shapes_72.to_string())
    print("\n=== Goal [7, 2] — Label counts ===")
    print(labels_72.to_string())

# %% [markdown]
# ## Goal location: [7, 11]
# %%
shapes_711, labels_711 = get_goal_dataframes('711', _pool=_pool)

# %%
if __name__ == "__main__":
    print("\n=== Goal [7, 11] — Shapes ===")
    print(shapes_711.to_string())
    print("\n=== Goal [7, 11] — Label counts ===")
    print(labels_711.to_string())

# %% [markdown]
# ## Goal location: [14, 7]
# %%
shapes_147, labels_147 = get_goal_dataframes('147', _pool=_pool)
_pool.shutdown(wait=False)

# %%
if __name__ == "__main__":
    print("\n=== Goal [14, 7] — Shapes ===")
    print(shapes_147.to_string())
    print("\n=== Goal [14, 7] — Label counts ===")
    print(labels_147.to_string())
