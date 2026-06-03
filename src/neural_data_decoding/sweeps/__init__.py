"""SLURM sweep dispatcher + ``.slurm`` template generator + parameter coverage."""

from neural_data_decoding.sweeps.dispatcher import (
    SWEEP_ENTRIES,
    SweepEntry,
    iter_by_choice,
    lookup,
    lookup_by_choice,
    total_sweep_count,
)
from neural_data_decoding.sweeps.slurm_template import (
    SlurmTemplateOptions,
    render_slurm_template,
    write_slurm_template,
)
from neural_data_decoding.sweeps.user_identity import (
    UserIdentity,
    identify_user,
    maybe_default_mail,
)

__all__ = [
    "SWEEP_ENTRIES",
    "SlurmTemplateOptions",
    "SweepEntry",
    "UserIdentity",
    "identify_user",
    "iter_by_choice",
    "lookup",
    "lookup_by_choice",
    "maybe_default_mail",
    "render_slurm_template",
    "total_sweep_count",
    "write_slurm_template",
]
