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
) -> Dict[str, torch.Tensor]:
    """Central DP FedAvg: aggregate only dp_param_keys via clipping + Gaussian noise.

    - Computes deltas per client: (local - global) for selected keys.
    - Clips each client's update vector to L2 norm <= clip_norm.
    - Sums clipped updates, adds Gaussian noise N(0, (sigma*clip_norm)^2 I), then averages.
    - Returns new global_state for dp_param_keys; other keys remain as in original global.
    """
    dp_keys = [k for k in dp_param_keys]
    n = len(client_states)
    if n == 0:
        return copy.deepcopy(global_state)

    # Build flattened deltas and shapes
    named_global = [(k, global_state[k].detach()) for k in dp_keys]
    _, shapes = _flatten_params(named_global, device)
    deltas = []
    for cs in client_states:
        named_local = [(k, cs[k].detach()) for k in dp_keys]
        local_vec, _ = _flatten_params(named_local, device)
        global_vec, _ = _flatten_params(named_global, device)
        deltas.append(local_vec - global_vec)

    # Clip per-client updates
    clipped = []
    clip_norm = float(clip_norm)
    for d in deltas:
        norm = torch.norm(d, p=2)
        scale = min(1.0, clip_norm / (norm + 1e-12))
        clipped.append(d * scale)

    # Sum, add noise, and average
    agg = torch.stack(clipped, dim=0).sum(dim=0)
    sigma = float(noise_multiplier)
    if sigma > 0 and clip_norm > 0:
        noise = torch.normal(mean=0.0, std=sigma * clip_norm, size=agg.shape, device=device)
        agg = agg + noise
    agg = agg / float(n)

    # New global = old global + aggregated delta
    global_vec, _ = _flatten_params(named_global, device)
    new_vec = global_vec + agg
    new_dp_state = _unflatten_to(shapes, new_vec)

    new_state = copy.deepcopy(global_state)
    for k, v in new_dp_state.items():
        new_state[k] = v.clone().detach()
    return new_state

