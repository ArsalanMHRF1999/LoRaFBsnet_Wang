import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
from Train_video.helper_fn import calculate_weight_updates, collect_probe_activity, get_learnable_weight_snapshot, collect_weight_statistics
import os

def train_video_model(model,
                      train_loader,
                      val_loader,
                      lr=3e-4, epochs=20,
                      device='cpu',
                      save_path='./best_model_feature_extractor_for_video.pth',
                      patience=5,
                      overfit_patience=2,
                      min_delta=1e-3
                      ):
    criterion = nn.CrossEntropyLoss(weight=None)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    model.to(device)
    best_val_acc = -1.0
    best_train_acc = -1.0
    best_gap = float('inf')
    best_wts = None
    epochs_without_improvement = 0
    consecutive_overfit_epochs = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'gap': [],
        'weight_stats': [],
        'activity_stats': [],
        "weight_updates": [],
        "lr": [],
        "overfit_signal": []
    }
    stop_reason = "Maximum epochs reached"
    previous_weights = get_learnable_weight_snapshot(model)

    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)
    print("Model Training Starting...")
    for epoch in range(epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        train_true, train_pred = [], []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            X = X.permute(1, 0, 2, 3, 4).contiguous()
            optimizer.zero_grad()
            out = model(X)
            logits = out.mean(0)
            loss = criterion(logits, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            preds = logits.argmax(1)
            train_true.extend(Y.detach().cpu().numpy())
            train_pred.extend(preds.detach().cpu().numpy())

        train_acc = accuracy_score(train_true, train_pred)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                X_val = X_val.permute(1, 0, 2, 3, 4).contiguous()
                out = model(X_val)
                logits = out.mean(0)
                preds = logits.argmax(1)
                y_true.extend(Y_val.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = accuracy_score(y_true, y_pred)
        gap = abs(train_acc - val_acc)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        current_weights = get_learnable_weight_snapshot(model)
        epoch_updates = calculate_weight_updates(previous_weights, current_weights)
        history["weight_updates"].append(epoch_updates)
        previous_weights = current_weights
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['gap'].append(gap)
        history["lr"].append(current_lr)
        weight_stats = collect_weight_statistics(model)
        history["weight_stats"].append(weight_stats)
        activity_stats = collect_probe_activity(model, val_loader)
        history["activity_stats"].append(activity_stats)

        # Overfitting Detection
        significant_improvement = val_acc > best_val_acc + min_delta
        overfit_signal = best_val_acc >= 0 and train_acc > best_train_acc + min_delta and val_acc < best_val_acc - min_delta
        if significant_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if overfit_signal:
            consecutive_overfit_epochs += 1
        else:
            consecutive_overfit_epochs = 0

        history['overfit_signal'] = overfit_signal

        is_better = (
                (val_acc > best_val_acc) or
                (val_acc == best_val_acc and gap < best_gap) or
                (val_acc == best_val_acc and gap ==
                 best_gap and train_acc > best_train_acc)
        )

        if is_better:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_gap = gap
            best_wts = {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_wts,
                'best_val_acc': best_val_acc,
                'best_train_acc': best_train_acc,
                'best_gap': best_gap,
                'history': history
            }, save_path)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"gap={gap:.4f} | "
            f"lr={current_lr:.2e}"
            f"no_improves={epochs_without_improvement} | "
            f"overfit={consecutive_overfit_epochs}"
        )

        if consecutive_overfit_epochs >= overfit_patience:
            stop_reason = (
                f"Overfitting detected for "
                f"{overfit_patience} consecutive epochs"
            )
            print(f"\nEarly stopping: {stop_reason}")
            break

        if epochs_without_improvement >= patience:
            stop_reason = (
                f"Validation accuracy did not improve "
                f"significantly for {patience} epochs"
            )
            print(f"\nEarly stopping: {stop_reason}")
            break

    if best_wts is not None:
        model.load_state_dict(best_wts)

    print(
        f"Best model | val_acc={best_val_acc:.4f} | "
        f"train_acc={best_train_acc:.4f} | gap={best_gap:.4f} | "
        f"saved → {save_path}"
    )
    print(f"Stop Reason: {stop_reason}")

    return model, history