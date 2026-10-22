import torch
import torch.nn as nn

def reset_monitoring_state(model):

    recurrent_layers = {
        1: model.recurrent_layer1,
        2: model.recurrent_layer2,
        3: model.recurrent_layer3,
        4: model.recurrent_layer4
    }

    for module in recurrent_layers.values():
        module.last_recurrent = None
        module.last_output = None
        module.last_external_feedback = None

    for module in model.feedback_connections.values():
        module.last_output = None


# ========================================================
# Timestep Tracker
# ========================================================

def collect_runtime_activity(model):
    stats = {
        "recurrent": {},
        "feedback": {}
    }
    recurrent_layers = {
        1: model.recurrent_layer1,
        2: model.recurrent_layer2,
        3: model.recurrent_layer3,
        4: model.recurrent_layer4
    }
    for idx, module in recurrent_layers.items():
        if module.last_recurrent is not None:
            x = module.last_recurrent
            stats["recurrent"][f"L{idx}"] = {
                "mean_abs": x.abs().mean().item(),
                "rms": x.pow(2).mean().sqrt().item(),
                "max": x.abs().max().item(),
                "active_ratio": ((x.abs() > 0.01).float().mean().item())
            }
    for name, module in model.feedback_connections.items():
        if module.last_output is None:
            continue
        x = module.last_output
        stats["feedback"][name] = {
            "mean_abs": x.abs().mean().item(),
            "rms": x.pow(2).mean().sqrt().item(),
            "max": x.abs().max().item(),
            "active_ratio": ((x.abs() > 0.01).float().mean().item())
        }
    return stats

def get_learnable_weight_snapshot(model):
    snapshot = {}
    recurrent_layers = {
        1: model.recurrent_layer1,
        2: model.recurrent_layer2,
        3: model.recurrent_layer3,
        4: model.recurrent_layer4
    }

    for idx, module in recurrent_layers.items():
        for name in ["dwconv", "pwconv"]:
            layer_module = getattr(module, name, None)
            if layer_module is None:
                continue
            if not hasattr(layer_module, "weight"):
                continue
            snapshot[f"L{idx}.{name}"] = (
                layer_module.weight.detach().cpu().clone()
            )

    for edge_name, module in model.feedback_connections.items():
        for sub_name, sub_module in module.named_modules():
            if not isinstance(sub_module, (nn.Conv2d, nn.ConvTranspose2d)):
                continue
            key = f"{edge_name}.{sub_name}.weight"
            snapshot[key] = (sub_module.weight.detach().cpu().clone())
    return snapshot

def collect_weight_statistics(model):
    stats = {
        "recurrent": {},
        "feedback": {}
    }
    recurrent_layers = {
        1: model.recurrent_layer1,
        2: model.recurrent_layer2,
        3: model.recurrent_layer3,
        4: model.recurrent_layer4
    }
    for idx, module in recurrent_layers.items():
        for name in ["dwconv", "pwconv"]:
            layer_module = getattr(module, name, None)
            if layer_module is None:
                continue
            if not hasattr(layer_module, "weight"):
                continue
            w = layer_module.weight.detach()
            stats["recurrent"][f"L{idx}.{name}"] = {
                "mean": w.mean().item(),
                "std": w.std().item(),
                "norm": w.norm().item(),
                "mean_abs": w.abs().mean().item(),
                "max_abs": w.abs().max().item()
            }

    for name, module in model.feedback_connections.items():
        for sub_name, sub_module in module.named_modules():
            if not isinstance(sub_module, (nn.Conv2d, nn.ConvTranspose2d)):
                continue
            if not hasattr(sub_module, "weight"):
                continue
            w = sub_module.weight.detach()
            key = name
            if sub_name:
                key += "." + sub_name
            stats["feedback"][key] = {
                "mean": w.mean().item(),
                "std": w.std().item(),
                "norm": w.norm().item(),
                "mean_abs": w.abs().mean().item(),
                "max_abs": w.abs().max().item()
            }
    return stats

def calculate_weight_updates(previous, current):
    updates = {
        "recurrent": {},
        "feedback": {}
    }
    for name, current_weight in current.items():
        if name not in previous:
            continue
        previous_weight = previous[name]
        delta = current_weight - previous_weight
        update_norm = delta.norm().item()
        current_norm = current_weight.norm().item()
        relative_update = update_norm / (previous_weight.norm().item() + 1e-12)
        stats = {
            "update_norm": update_norm,
            "relative_update": relative_update,
            "current_norm": current_norm
        }
        if name.startswith("L"):
            updates["recurrent"][name] = stats
        else:
            updates["feedback"][name] = stats
    return updates

@torch.no_grad()
def collect_probe_activity(model, loader, device):
    model.eval()
    X, Y = next(iter(loader))
    X = X.to(device)
    X = X.permute(1, 0, 2, 3, 4).contiguous()
    reset_monitoring_state(model)
    _ = model(X)
    stats = collect_runtime_activity(model)
    return stats