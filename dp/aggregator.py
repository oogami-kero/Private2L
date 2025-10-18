import copy
from typing import Dict, Iterable, Tuple, List
import torch


def _flatten_params(named_tensors: List[Tuple[str, torch.Tensor]], device: torch.device) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
    flat = []
    shapes = []
    for k, t in named_tensors:
        tt = t.detach().to(device)
        shapes.append((k, tuple(tt.shape)))
        flat.append(tt.reshape(-1))
    if flat:
        return torch.cat(flat), shapes
    return torch.zeros(0, device=device), shapes


def _unflatten_to(named_shapes: List[Tuple[str, Tuple[int, ...]]], vec: torch.Tensor) -> Dict[str, torch.Tensor]:
    state = {}
    offset = 0
    for k, shape in named_shapes:
        n = 1
        for s in shape:
            n *= int(s)
        if n > 0:
            state[k] = vec[offset:offset + n].view(shape)
        else:
            state[k] = torch.zeros(shape, dtype=vec.dtype, device=vec.device)
        offset += n
    return state


def clip_and_aggregate(
    client_states: List[Dict[str, torch.Tensor]],
    global_state: Dict[str, torch.Tensor],
    dp_param_keys: Iterable[str],
    clip_norm: float,
    noise_multiplier: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[float]]]:
    """Central DP FedAvg: aggregate only dp_param_keys via clipping + Gaussian noise.

    - Computes deltas per client: (local - global) for selected keys.
    - Clips each client's update vector to L2 norm <= clip_norm.
    - Sums clipped updates, adds Gaussian noise N(0, (sigma*clip_norm)^2 I), then averages.
    - Returns the updated global_state for dp_param_keys and telemetry containing
      each client's pre-clipping norm and the applied scale factor.
    """
    dp_keys = [k for k in dp_param_keys]
    n = len(client_states)
    if n == 0:
        telemetry = {
            'pre_clip_norms': [],
            'clip_scales': [],
            'clip_norm': float(clip_norm),
        }
        return copy.deepcopy(global_state), telemetry

    # Build flattened deltas and shapes
    named_global = [(k, global_state[k].detach()) for k in dp_keys]
    global_vec, shapes = _flatten_params(named_global, device)
    deltas = []
    for cs in client_states:
        named_local = [(k, cs[k].detach()) for k in dp_keys]
        local_vec, _ = _flatten_params(named_local, device)
        deltas.append(local_vec - global_vec)

    # Clip per-client updates
    clipped = []
    clip_norm = float(clip_norm)
    pre_clip_norms: List[float] = []
    clip_scales: List[float] = []
    for d in deltas:
        norm = torch.norm(d, p=2)
        norm_item = float(norm.item())
        if clip_norm > 0:
            scale = min(1.0, clip_norm / (norm_item + 1e-12))
        else:
            scale = 1.0 if norm_item == 0.0 else 0.0
        pre_clip_norms.append(norm_item)
        clip_scales.append(float(scale))
        clipped.append(d * scale)

    # Sum, add noise, and average
    agg = torch.stack(clipped, dim=0).sum(dim=0)
    sigma = float(noise_multiplier)
    if sigma > 0 and clip_norm > 0:
        noise = torch.normal(mean=0.0, std=sigma * clip_norm, size=agg.shape, device=device)
        agg = agg + noise
    agg = agg / float(n)

    # New global = old global + aggregated delta
    new_vec = global_vec + agg
    new_dp_state = _unflatten_to(shapes, new_vec)

    new_state = copy.deepcopy(global_state)
    for k, v in new_dp_state.items():
        new_state[k] = v.clone().detach()
    telemetry = {
        'pre_clip_norms': pre_clip_norms,
        'clip_scales': clip_scales,
        'clip_norm': clip_norm,
    }
    return new_state, telemetry

