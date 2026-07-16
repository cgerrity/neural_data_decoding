# Add a new loss component

A new loss term plugs into the multi-objective aggregator — you write a kernel
and thread it through `aggregate_normalized_losses`. The EMA normalization then
balances it against the existing components automatically.

## Steps

1. **Write the kernel** in `training/losses/` (e.g.
   `temporal_smoothness.py`). It takes tensors and returns a scalar loss,
   differentiable w.r.t. its inputs. Follow the existing kernels
   (`elbo.py`, `classification.py`, `confidence.py`) for style — NumPy-style
   docstrings, NaN-safe reductions.

2. **Add a keyword argument** to `aggregate_normalized_losses` in
   `training/losses/multi_objective.py`:

   ```python
   def aggregate_normalized_losses(
       *,
       reconstruction_loss=None,
       ...,
       temporal_smoothness_loss=None,     # <- new
       weights, priors, ...,
   ):
   ```

3. **Give it an EMA prior** — add a `temporal_smoothness` field to `LossPriors`
   and process it with `_process_component` like the others (normalize →
   rescale → weight).

4. **Route it into a sub-total** — add it to `Loss_Decoder` or `Loss_Classifier`
   ([Multi-objective losses](../concepts/multi_objective_losses.md)) so its
   gradient reaches the right subnetwork.

5. **Add a weight key** — `"temporal_smoothness"` in the `weights` dict, and
   (optionally) in the curriculum's `make_weight_schedule` so it can be
   scheduled.

## Why the normalization matters

Because each component is divided by its own running magnitude before summing,
you **don't hand-tune your loss's raw scale** — only its *weight* (relative
importance). A component 1000× larger than classification won't dominate; the
weight ratio governs the balance.

## Gotchas

- A `NaN` weight means "skip from the gradient sum"; `0.0` means "present but
  zero." If your loss has no effect, check the weight isn't `NaN`.
- Make sure the term is actually summed into a sub-total — an unrouted kwarg is
  silently dropped.

## Related

- Notebook `notebooks/06_loss_orchestration/06.11_single_total_loss_three_subnetworks.ipynb`.
