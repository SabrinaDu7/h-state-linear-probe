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


def _load_hs_flat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load hidden_states and labels from a trajectories.pt file.

    Returns:
        hs_flat  : float32 array of shape (N, hidden_dim), N = B*T
        lbl_flat : int/float array of shape (N,)
    """
    class _Unpickler(pickle.Unpickler):
        def __init__(self, f):
            super().__init__(f)
            self._storages: dict[int, tuple[str, np.dtype]] = {}

        def persistent_load(self, pid):
            _, storage_cls, key, _location, _numel = pid
            storage = (key, _NP_DTYPE_MAP.get(storage_cls.__name__, np.float32))
            token = object()
            self._storages[id(token)] = storage
            return token

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

    def _read_tensor(zf: zipfile.ZipFile, prefix: str, meta: dict, storages: dict) -> np.ndarray:
        key, dtype = storages[meta['__sid__']]
        with zf.open(f'{prefix}/data/{key}') as f:
            raw = np.frombuffer(f.read(), dtype=dtype)
        numel = int(np.prod(meta['__size__']))
        return raw[meta['__offset__']: meta['__offset__'] + numel].reshape(meta['__size__'])

    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split('/')[0]
        with zf.open(f'{prefix}/data.pkl') as f:
            up = _Unpickler(f)
            result = up.load()
        hs  = _read_tensor(zf, prefix, result['hidden_states'], up._storages)  # (B, T, 500)
        lbl = _read_tensor(zf, prefix, result['labels'],        up._storages)  # (B, T)

    return hs.reshape(-1, hs.shape[-1]), lbl.reshape(-1)


def _norm_stats(hs_flat: np.ndarray, lbl_flat: np.ndarray) -> dict[str, float]:
    """L2 norm of the mean hidden-state vector for all / label-0 / label-1."""
    def _norm(arr: np.ndarray) -> float:
        return float(np.linalg.norm(arr.mean(axis=0)))
    return {
        'norm_all':    _norm(hs_flat),
        'norm_label0': _norm(hs_flat[lbl_flat == 0]),
        'norm_label1': _norm(hs_flat[lbl_flat == 1]),
    }


def _load_hs_stats(path: Path) -> dict[str, float]:
    """
    Norm of the mean hidden-state vector (all 500 dims) for all / label-0 / label-1.
    """
    hs_flat, lbl_flat = _load_hs_flat(path)
    return _norm_stats(hs_flat, lbl_flat)


def _load_hs_topk_stats(path: Path, k: int = 100) -> dict[str, float]:
    """
    Select the top-k dimensions by overall std, then compute the norm of the
    mean hidden-state vector restricted to those dimensions.

    Top-k indices are chosen from all samples (class-agnostic), then the same
    indices are applied when filtering by label.
    """
    hs_flat, lbl_flat = _load_hs_flat(path)
    top_idx = np.argsort(hs_flat.std(axis=0))[-k:]   # (k,) indices of highest-std dims
    hs_topk = hs_flat[:, top_idx]                     # (N, k)
    return _norm_stats(hs_topk, lbl_flat)


def _goal_dirs(goal: str, data_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"data_cur_lroom_step(\d+)_goal{re.escape(goal)}$")
    return sorted(
        [(int(m.group(1)), d) for d in data_dir.iterdir() if (m := pattern.match(d.name))],
        key=lambda x: x[0],
    )


def get_hs_stats_dataframe(
    goal: str,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Norm of the mean hidden-state vector (all 500 dims) across step checkpoints.

      index   : ['norm_all', 'norm_label0', 'norm_label1']
      columns : step counts (sorted ascending)
    """
    stats_data = {step: _load_hs_stats(d / 'trajectories.pt') for step, d in _goal_dirs(goal, data_dir)}
    return pd.DataFrame(stats_data).rename_axis('stat')


def get_hs_topk_stats_dataframe(
    goal: str,
    data_dir: Path = DATA_DIR,
    k: int = 100,
) -> pd.DataFrame:
    """
    Norm of the mean hidden-state vector restricted to the top-k highest-std
    dimensions across step checkpoints.

    Top-k dimensions are selected per checkpoint by overall (class-agnostic) std.

      index   : ['norm_all', 'norm_label0', 'norm_label1']
      columns : step counts (sorted ascending)
    """
    stats_data = {
        step: _load_hs_topk_stats(d / 'trajectories.pt', k=k)
        for step, d in _goal_dirs(goal, data_dir)
    }
    return pd.DataFrame(stats_data).rename_axis('stat')


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
hs_stats_72 = get_hs_stats_dataframe('72')
hs_topk_stats_72 = get_hs_topk_stats_dataframe('72')

# %%
if __name__ == "__main__":
    print("=== Goal [7, 2] — Shapes ===")
    print(shapes_72.to_string())
    print("\n=== Goal [7, 2] — Label counts ===")
    print(labels_72.to_string())
    print("\n=== Goal [7, 2] — Hidden State Stats ===")
    print(hs_stats_72.to_string())
    print("\n=== Goal [7, 2] — Hidden State Top-100 Std Stats ===")
    print(hs_topk_stats_72.to_string())

# %% [markdown]
# ## Goal location: [7, 11]
# %%
shapes_711, labels_711 = get_goal_dataframes('711', _pool=_pool)
hs_stats_711 = get_hs_stats_dataframe('711')
hs_topk_stats_711 = get_hs_topk_stats_dataframe('711')

# %%
if __name__ == "__main__":
    print("\n=== Goal [7, 11] — Shapes ===")
    print(shapes_711.to_string())
    print("\n=== Goal [7, 11] — Label counts ===")
    print(labels_711.to_string())
    print("\n=== Goal [7, 11] — Hidden State Stats ===")
    print(hs_stats_711.to_string())
    print("\n=== Goal [7, 11] — Hidden State Top-100 Std Stats ===")
    print(hs_topk_stats_711.to_string())

# %% [markdown]
# ## Goal location: [14, 7]
# %%
shapes_147, labels_147 = get_goal_dataframes('147', _pool=_pool)
hs_stats_147 = get_hs_stats_dataframe('147')
hs_topk_stats_147 = get_hs_topk_stats_dataframe('147')
_pool.shutdown(wait=False)

# %%
if __name__ == "__main__":
    print("\n=== Goal [14, 7] — Shapes ===")
    print(shapes_147.to_string())
    print("\n=== Goal [14, 7] — Label counts ===")
    print(labels_147.to_string())
    print("\n=== Goal [14, 7] — Hidden State Stats ===")
    print(hs_stats_147.to_string())
    print("\n=== Goal [14, 7] — Hidden State Top-100 Std Stats ===")
    print(hs_topk_stats_147.to_string())


# %% [markdown]
# ## Verify LOGO
# %%
"""Quick sanity-check: print which samples land in each LOGO val fold."""
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from probe import load_and_flatten
from sklearn.model_selection import LeaveOneGroupOut

# %%
cfg = Config(dataset_step=1208, dataset_goals=[72, 711, 147], logo=True)

print(f"Loading data for goals {cfg.dataset_goals} at step {cfg.dataset_step}...\n")
X, y, groups, goal_ids = load_and_flatten(cfg.data_paths, cfg.dataset_goals)

splits = list(LeaveOneGroupOut().split(X, y, goal_ids))
fold_labels = [int(cfg.dataset_goals[int(np.unique(goal_ids[val_idx])[0])]) for _, val_idx in splits]

print(f"Total samples: {len(X)}  |  goal_ids unique values: {np.unique(goal_ids)}\n")

for fold_label, (train_idx, val_idx) in zip(fold_labels, splits):
    val_goals = np.unique(goal_ids[val_idx])
    train_goals = np.unique(goal_ids[train_idx])
    val_label_counts = Counter(y[val_idx].tolist())
    train_label_counts = Counter(y[train_idx].tolist())

    print(f"--- Held-out goal: {fold_label} ---")
    print(f"  val   indices: {len(val_idx):>6}  |  goal_ids in val:   {val_goals}  |  labels: {dict(val_label_counts)}")
    print(f"  train indices: {len(train_idx):>6}  |  goal_ids in train: {train_goals}  |  labels: {dict(train_label_counts)}")
    print()
