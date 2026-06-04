"""Start-of-run banner — Python analog of ``cgg_runAutoEncoder.m`` lines 320-323.

The MATLAB pipeline prints three diagnostic blocks at the top of every
run:

1. ``disp(cfg_Encoder)`` — full config dump
2. ``disp(datetime)`` — wall-clock timestamp
3. ``gpuDeviceTable([...])`` — Index/Name/Memory/etc. of available GPUs

Plus ``cgg_assignSLURMSession.m`` line 23's
``>>> Current SLURM Aim is Base Case - Fold N - Session SSS`` line
that lands in the SLURM output file.

The Python equivalent collects the same diagnostic surface plus a few
items MATLAB doesn't have (git SHA + branch, identified user). All
lines go to a single string so:

* CLI can print to stderr early in `_cmd_train`
* Tests can pin the rendering without subprocess-launching anything
* A future structured-log target can swap the string formatter
  without changing collection

The collection function is intentionally tolerant: a missing GPU, no
git repo, or no identified user all produce a "n/a"-style cell rather
than raising. Run banners are diagnostic, not load-bearing.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_data_decoding.sweeps.user_identity import UserIdentity, identify_user


# Cfg keys highlighted in the banner header so the user can sanity-check
# at a glance without scrolling through the full dump. Mirrors what
# someone debugging a sweep entry would actually look at first.
_CFG_HEADLINE_KEYS: tuple[str, ...] = (
    "epoch",
    "target",
    "model_name",
    "classifier_name",
    "is_variational",
    "optimizer",
    "weighted_loss",
    "loss_type_decoder",
    "multiple_instance_learning_type",
    "dynamic_parameter_set",
    "num_epochs_full",
    "num_epochs_autoencoder",
    "mini_batch_size",
    "initial_learning_rate",
    "hidden_sizes",
    "classifier_hidden_size",
    "data_width",
    "window_stride",
    "subset",
    "fold",
)

_BANNER_RULE = "=" * 78
_BANNER_SUBRULE = "-" * 78


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """Minimal GPU descriptor for the banner."""

    index: int
    name: str
    total_memory_gb: float
    is_selected: bool


@dataclass(frozen=True, slots=True)
class GitState:
    """Read-only snapshot of the local git state (sha + branch)."""

    sha: str
    branch: str

    @property
    def short_sha(self) -> str:
        """First 12 characters of the SHA (or empty)."""
        return self.sha[:12] if self.sha else ""


@dataclass(frozen=True, slots=True)
class RunBannerData:
    """Everything the banner needs — collected once, formatted in pure code.

    Splitting collection (impure: env, GPU, git) from formatting (pure)
    makes the renderer trivially testable: pass a hand-built instance
    and assert on the string.
    """

    timestamp_utc: str
    user: UserIdentity
    git: GitState
    config_name: str
    sweep_index: int | None
    sweep_description: str | None
    sweep_notes: tuple[str, ...]
    session_run_idx: int | None
    fold: int
    subset_label: str
    result_dir: Path
    cfg_headline: dict[str, Any]
    use_real_data: bool
    num_train_trials: int
    num_val_trials: int
    num_test_trials: int
    num_classes_per_dim: list[int]
    sample_shape: tuple[int, ...]
    torch_version: str
    gpus: tuple[GpuInfo, ...]


def collect_banner_data(
    *,
    config_name: str,
    cfg: Any,
    args: Any,
    result_dir: Path,
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    use_real_data: bool,
    num_classes_per_dim: list[int],
) -> RunBannerData:
    """Gather every banner field from live process state.

    Tolerant by design — every accessor degrades to a sensible
    empty/default rather than raising. Run banners are diagnostic,
    not load-bearing.
    """
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    user = identify_user()
    git = _read_git_state()
    sample_shape = _probe_sample_shape(train_ds)
    gpus = _collect_gpu_info()
    torch_version = _safe_torch_version()

    headline: dict[str, Any] = {}
    for key in _CFG_HEADLINE_KEYS:
        try:
            value = cfg.get(key, "<unset>")
        except Exception:
            value = "<error>"
        headline[key] = value

    sweep_idx = getattr(args, "sweep_index", None)
    sweep_description = (
        getattr(cfg, "sweep_description", None) if sweep_idx is not None else None
    )
    sweep_notes_attr = getattr(args, "sweep_notes", None)
    sweep_notes = tuple(sweep_notes_attr) if sweep_notes_attr else ()

    session_run_idx = getattr(args, "session_run_idx", None)
    subset_label = _format_subset_label(cfg)

    return RunBannerData(
        timestamp_utc=timestamp,
        user=user,
        git=git,
        config_name=config_name,
        sweep_index=int(sweep_idx) if sweep_idx is not None else None,
        sweep_description=sweep_description,
        sweep_notes=sweep_notes,
        session_run_idx=(
            int(session_run_idx) if session_run_idx is not None else None
        ),
        fold=int(getattr(cfg, "fold", 0)),
        subset_label=subset_label,
        result_dir=result_dir,
        cfg_headline=headline,
        use_real_data=use_real_data,
        num_train_trials=len(train_ds),
        num_val_trials=len(val_ds),
        num_test_trials=len(test_ds),
        num_classes_per_dim=list(num_classes_per_dim),
        sample_shape=sample_shape,
        torch_version=torch_version,
        gpus=tuple(gpus),
    )


def render_banner(data: RunBannerData) -> str:
    """Render :class:`RunBannerData` as a multi-line banner string."""
    lines: list[str] = [
        _BANNER_RULE,
        f">>> neural_data_decoding run — {data.timestamp_utc}",
        _BANNER_RULE,
    ]

    # Identity + provenance
    user_label = (
        f"{data.user.username or '?'}"
        + (" [Charles auto-detected]" if data.user.is_charles else "")
    )
    git_label = (
        f"{data.git.short_sha} ({data.git.branch})"
        if data.git.short_sha
        else "<not a git repository>"
    )
    lines.append(f"user      : {user_label}")
    lines.append(f"git       : {git_label}")
    lines.append(f"torch     : {data.torch_version}")
    lines.append(_BANNER_SUBRULE)

    # Run identity
    lines.append(f"config    : {data.config_name}")
    if data.sweep_index is not None:
        desc = data.sweep_description or "<no description>"
        lines.append(f"sweep idx : {data.sweep_index} — {desc}")
        for note in data.sweep_notes:
            lines.append(f"            note: {note}")
    if data.session_run_idx is not None:
        lines.append(f"sessionRun: {data.session_run_idx}")
    lines.append(f"fold      : {data.fold}")
    lines.append(f"subset    : {data.subset_label}")
    lines.append(f"result_dir: {data.result_dir}")
    lines.append(_BANNER_SUBRULE)

    # Dataset + shape
    data_kind = "real (.mat)" if data.use_real_data else "synthetic"
    lines.append(f"dataset   : {data_kind}")
    lines.append(
        f"trials    : train={data.num_train_trials} "
        f"val={data.num_val_trials} test={data.num_test_trials}"
    )
    if data.sample_shape:
        shape_str = "x".join(str(d) for d in data.sample_shape)
        lines.append(f"sample    : ({shape_str}) (W, T, A, C)")
    lines.append(f"classes/d : {data.num_classes_per_dim}")
    lines.append(_BANNER_SUBRULE)

    # Cfg headline
    lines.append("cfg headline:")
    width = max(len(k) for k in data.cfg_headline)
    for key, value in data.cfg_headline.items():
        lines.append(f"  {key:<{width}} = {value!r}")
    lines.append(_BANNER_SUBRULE)

    # GPU table — MATLAB's gpuDeviceTable analog
    if not data.gpus:
        lines.append("GPUs      : <none detected — CPU only>")
    else:
        lines.append("GPUs      :")
        lines.append(
            f"  {'idx':>3} {'name':<32} {'mem_GB':>8} {'selected':>9}"
        )
        for g in data.gpus:
            lines.append(
                f"  {g.index:>3} {g.name[:32]:<32} {g.total_memory_gb:>8.2f} "
                f"{'yes' if g.is_selected else '':>9}"
            )
    lines.append(_BANNER_RULE)
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Tolerant collectors (impure side)
# ----------------------------------------------------------------------


def _read_git_state(cwd: Path | None = None) -> GitState:
    """Best-effort SHA + branch lookup; ``GitState('', '')`` on failure."""
    sha = _git_one_line(["git", "rev-parse", "HEAD"], cwd=cwd)
    branch = _git_one_line(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    return GitState(sha=sha, branch=branch)


def _git_one_line(cmd: list[str], *, cwd: Path | None) -> str:
    """Run ``cmd`` and return the first line of stdout, or "" on any failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def _collect_gpu_info() -> list[GpuInfo]:
    """Return one :class:`GpuInfo` per visible CUDA device, or ``[]``."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out: list[GpuInfo] = []
    try:
        selected = torch.cuda.current_device()
    except RuntimeError:
        selected = -1
    for idx in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(idx)
            name = props.name
            mem_gb = props.total_memory / (1024 ** 3)
        except (RuntimeError, AttributeError):
            name = "<error>"
            mem_gb = 0.0
        out.append(
            GpuInfo(
                index=idx,
                name=name,
                total_memory_gb=mem_gb,
                is_selected=(idx == selected),
            )
        )
    return out


def _safe_torch_version() -> str:
    """Return the running ``torch.__version__`` or ``"?"`` on failure."""
    try:
        import torch
    except ImportError:
        return "?"
    return getattr(torch, "__version__", "?")


def _probe_sample_shape(ds: Any) -> tuple[int, ...]:
    """Return the shape of ``ds[0]``'s feature tensor as a plain tuple."""
    try:
        x = ds[0][0]
    except (IndexError, KeyError, TypeError):
        return ()
    shape = getattr(x, "shape", ())
    return tuple(int(d) for d in shape)


def _format_subset_label(cfg: Any) -> str:
    """Render the cfg.subset field in a banner-friendly way."""
    try:
        subset = cfg.get("subset", True)
    except Exception:
        return "<error>"
    if isinstance(subset, bool):
        return "all sessions" if not subset else "single-session (filter-driven)"
    s = str(subset)
    if not s or s.lower() == "all":
        return "all sessions"
    return f"single session: {s}"


__all__ = [
    "GitState",
    "GpuInfo",
    "RunBannerData",
    "collect_banner_data",
    "render_banner",
]
