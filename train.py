# train.py
import logging
import os
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import random

# Import model classes
from model import MLPBaseline, AdvancedAttentionModel, HybridAttentionModel, ImageOnlyBaseline


def train_model(model, model_save_path, train_loader, val_loader, optimizer, scheduler, ce_criterion, anat_criterion,
                contrastive_criterion, args, device, contrastive_weight=0.1):
    """(V19) Main training loop, supports multimodal input."""
    model.to(device)
    best_val_acc = 0.0

    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    if anat_criterion and hasattr(anat_criterion, 'label_classes_') and hasattr(train_loader.dataset, 'tensors'):
        try:
            pass
        except Exception as e:
            logging.warning(f"Could not set label_classes_ for AnatomicalLoss: {e}")

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_ce, total_anat, total_cont = 0, 0, 0, 0

        current_anat_weight = args.anatomical_weight_final * (1 + math.cos(math.pi * epoch / args.epochs)) / 2

        for keypoint_inputs, image_features, labels in train_loader:
            keypoint_inputs = keypoint_inputs.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)

            anat_loss = torch.tensor(0.0, device=device)
            if anat_criterion and current_anat_weight > 0:
                anat_loss = current_anat_weight * anat_criterion(keypoint_inputs, labels)

            augmented_keypoints = keypoint_inputs.clone()
            augmented_image_features = image_features.clone()

            if torch.rand(1).item() < 0.8:
                if torch.rand(1).item() < 0.5:
                    augmented_keypoints = augment_spatial(augmented_keypoints)
                else:
                    if torch.rand(1).item() < 0.7:
                        augmented_keypoints += torch.randn_like(augmented_keypoints) * 0.5
                    else:
                        augmented_keypoints = occlude_points(augmented_keypoints, num_occlusions=3)

                noise_std = 0.1
                augmented_image_features += torch.randn_like(augmented_image_features) * noise_std

            optimizer.zero_grad(set_to_none=True)
            outputs = model(augmented_keypoints, augmented_image_features)

            if isinstance(outputs, tuple) or isinstance(outputs, list):
                logits, embeddings = outputs[0], outputs[1]
            else:
                logits = outputs;
                embeddings = None

            loss = ce_criterion(logits, labels)
            total_ce += loss.item()

            loss += anat_loss
            total_anat += anat_loss.item() if isinstance(anat_loss, torch.Tensor) else anat_loss

            if contrastive_criterion and embeddings is not None and contrastive_weight > 0:
                cont_loss = contrastive_weight * contrastive_criterion(embeddings, labels)
                loss += cont_loss
                total_cont += cont_loss.item()

            loss.backward()
            if args.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_loss = evaluate_model(model, val_loader, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} "
            f"(CE: {total_ce / len(train_loader):.4f}, Anat: {total_anat / len(train_loader):.4f}, Cont: {total_cont / len(train_loader):.4f}) | "
            f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Anat W: {current_anat_weight:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"  -> New best validation accuracy: {best_val_acc:.4f}. Model saved to {model_save_path}")
        else:
            patience_counter += 1
            logging.info(f"  -> No improvement in validation accuracy for {patience_counter} epoch(s).")

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            logging.info(f"\n--- Early stopping triggered after {patience_counter} epochs with no improvement. ---")
            logging.info(f"Best validation accuracy achieved: {best_val_acc:.4f}")
            break

    return history


def evaluate_model(model, data_loader, device):
    """(V19) Evaluates model accuracy and loss (supports multimodal)."""
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for keypoint_inputs, image_features, labels in data_loader:
            keypoint_inputs = keypoint_inputs.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)

            outputs = model(keypoint_inputs, image_features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    avg_val_loss = total_val_loss / len(data_loader)
    all_labels_np = torch.cat(all_labels).numpy()
    all_preds_np = torch.cat(all_preds).numpy()
    acc = accuracy_score(all_labels_np, all_preds_np)

    return acc, avg_val_loss


def final_test_and_report(model_class, model_path, test_loader, device, label_encoder, args, input_dim, config,
                          output_paths, image_feature_dim=2048):
    """(V19) Loads model, evaluates, generates reports (supports multimodal)."""
    logging.info(f"\n--- Final Evaluation for {os.path.basename(model_path)} ---")
    num_classes = len(label_encoder.classes_)

    use_gated = config.get("use_gated_fusion", False)
    use_dynamic = config.get("use_dynamic_graph", False)
    current_embed_dim = config.get("embed_dim", args.embed_dim)

    # --- [ AI BUG FIX START ] ---
    # 从 config 字典中读取正确的超参数，而不是使用 args 默认值
    current_num_layers = config.get("num_layers", args.num_layers)
    current_num_heads = config.get("num_heads", args.num_heads)
    # --- [ AI BUG FIX END ] ---

    use_cross_attn = config.get("use_cross_attn", True)

    # --- [ FIX ] ---
    # Default must match the training loop in main.py
    use_gcn_branch = config.get("use_gcn_branch", False)
    # --- [ FIX END ] ---

    model_kwargs = {"input_dim": input_dim, "num_classes": num_classes,
                    "num_layers": current_num_layers,  # <-- 使用修正后的变量
                    "num_heads": current_num_heads,  # <-- 使用修正后的变量
                    "embed_dim": current_embed_dim, "dropout": args.dropout,
                    "image_feature_dim": image_feature_dim}

    # Instantiate the correct model
    if model_class == MLPBaseline:
        model_kwargs.pop("image_feature_dim")
        flat_kpt_dim = 21 * input_dim
        model = model_class(input_size=flat_kpt_dim, num_classes=num_classes)
        logging.info("Instantiated MLPBaseline model for evaluation.")
    elif model_class == AdvancedAttentionModel:
        model_kwargs.pop("image_feature_dim")
        model = model_class(**model_kwargs)
        logging.info("Instantiated AdvancedAttentionModel model for evaluation.")
    elif model_class == ImageOnlyBaseline:
        model = model_class(image_feature_dim=image_feature_dim,
                            num_classes=num_classes,
                            dropout=args.dropout)
        logging.info(f"Instantiated ImageOnlyBaseline (Image Dim: {image_feature_dim}) for evaluation.")
    elif model_class == HybridAttentionModel:
        model = model_class(**model_kwargs, use_gated_fusion=use_gated, use_dynamic_graph=use_dynamic,
                            use_cross_attn=use_cross_attn, use_gcn_branch=use_gcn_branch)
        gated_status = "ENABLED" if use_gated else "DISABLED"
        dynamic_status = "ENABLED" if use_dynamic else "DISABLED"
        cross_attn_status = "ENABLED" if use_cross_attn else "DISABLED (Late Fusion)"
        gcn_status = "ENABLED" if use_gcn_branch else "DISABLED"
        logging.info(
            f"Instantiated HybridAttentionModel (Multimodal Ready - Image Dim: {image_feature_dim}) "
            f"with Cross-Attn {cross_attn_status}, GCN {gcn_status}, Gated Fusion {gated_status}, Dynamic Graph {dynamic_status}.")
    else:
        logging.error(
            f"Unknown model class: {model_class.__name__ if hasattr(model_class, '__name__') else model_class}");
        return 0.0

    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        logging.error(f"ERROR: Model file not found at {model_path}.");
        return 0.0
    except Exception as e:
        logging.error(f"Error loading model weights: {e}");
        return 0.0

    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for keypoint_inputs, image_features, labels in test_loader:
            outputs = model(keypoint_inputs.to(device), image_features.to(device))
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    all_labels_np, all_preds_np = torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
    logging.info(
        f"\nClassification Report:\n{classification_report(all_labels_np, all_preds_np, target_names=label_encoder.classes_, digits=4, zero_division=0)}")
    cm = confusion_matrix(all_labels_np, all_preds_np)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix for {os.path.basename(model_path)}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_filename = f"cm_{os.path.basename(model_path).replace('.pth', '.png')}"
    cm_path = os.path.join(output_paths['cm_dir'], cm_filename)
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_path}")
    return accuracy_score(all_labels_np, all_preds_np)


def test_robustness(model_class, model_path, test_dataset, device, label_encoder, args, input_dim, config,
                    image_feature_dim=2048):
    """(V19) Robustness testing (supports multimodal)."""
    logging.info(f"\n--- Robustness Test for {os.path.basename(model_path)} ---")

    # [删除] 移除了跳过 MLP 的代码
    # if model_class == MLPBaseline:
    #     logging.info("Skipping robustness test for MLP.");
    #     return None

    num_classes = len(label_encoder.classes_)
    use_gated = config.get("use_gated_fusion", False)
    use_dynamic = config.get("use_dynamic_graph", False)
    current_embed_dim = config.get("embed_dim", args.embed_dim)
    use_cross_attn = config.get("use_cross_attn", True)

    # --- [ AI BUG FIX START ] ---
    # 从 config 字典中读取正确的超参数，而不是使用 args 默认值
    current_num_layers = config.get("num_layers", args.num_layers)
    current_num_heads = config.get("num_heads", args.num_heads)
    # --- [ AI BUG FIX END ] ---

    # --- [ FIX ] ---
    # Default must match the training loop in main.py
    use_gcn_branch = config.get("use_gcn_branch", False)
    # --- [ FIX END ] ---

    model_kwargs = {"input_dim": input_dim, "num_classes": num_classes,
                    "num_layers": current_num_layers,  # <-- 使用修正后的变量
                    "num_heads": current_num_heads,  # <-- 使用修正后的变量
                    "embed_dim": current_embed_dim, "dropout": args.dropout,
                    "image_feature_dim": image_feature_dim}

    # --- [ AI 补丁：添加了 MLPBaseline 的实例化逻辑 ] ---
    if model_class == MLPBaseline:
        model_kwargs.pop("image_feature_dim")
        flat_kpt_dim = 21 * input_dim
        model = model_class(input_size=flat_kpt_dim, num_classes=num_classes)
        logging.info(f"Instantiated MLPBaseline (Input Dim: {flat_kpt_dim}) for robustness test.")
    # --- [ AI 补丁结束 ] ---

    elif model_class == HybridAttentionModel:
        model = model_class(**model_kwargs, use_gated_fusion=use_gated, use_dynamic_graph=use_dynamic,
                            use_cross_attn=use_cross_attn, use_gcn_branch=use_gcn_branch)
        logging.info("Instantiated HybridAttentionModel for robustness test.")

    elif model_class == ImageOnlyBaseline:
        model = model_class(image_feature_dim=image_feature_dim,
                            num_classes=num_classes,
                            dropout=args.dropout)
        logging.info(f"Instantiated ImageOnlyBaseline (Image Dim: {image_feature_dim}) for robustness test.")

    elif model_class == AdvancedAttentionModel:
        model_kwargs.pop("image_feature_dim")
        model = model_class(**model_kwargs)
        logging.info("Instantiated AdvancedAttentionModel for robustness test.")

    else:
        logging.error(
            f"Unknown model class for robustness test: {model_class.__name__ if hasattr(model_class, '__name__') else model_class}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"ERROR: Model file not found at {model_path} during robustness test.");
        return None
    # [新增] 捕获由参数不匹配引起的 RuntimeError
    except RuntimeError as e:
        logging.error(f"ERROR: Failed to load model weights. This is likely an architecture mismatch: {e}")
        return None

    model.to(device)
    results = {}

    keypoint_test_tensor = test_dataset.tensors[0]
    image_features_tensor = test_dataset.tensors[1]
    y_test_tensor = test_dataset.tensors[2]

    noise_scenarios = {
        "gaussian_0.5": lambda x: x + torch.randn_like(x) * 0.5,
        "gaussian_1.0": lambda x: x + torch.randn_like(x) * 1.0,
        "occlusion_3": lambda x: occlude_points(x, num_occlusions=3),
        "structured_finger": lambda x: structured_noise(x)
    }

    if model_class == ImageOnlyBaseline:
        logging.info("ImageOnlyBaseline is not sensitive to keypoint noise, but running test for consistency.")

    # --- [ AI 补丁：添加了 MLP 的特别说明 ] ---
    if model_class == MLPBaseline:
        logging.info("MLPBaseline flattens keypoints; robustness test is running.")
    # --- [ AI 补丁结束 ] ---

    for name, noise_func in noise_scenarios.items():
        noisy_keypoints = noise_func(keypoint_test_tensor.clone())

        noisy_loader = DataLoader(
            TensorDataset(noisy_keypoints, image_features_tensor, y_test_tensor),
            batch_size=args.batch_size * 2
        )

        acc, _ = evaluate_model(model, noisy_loader, device)
        results[name] = acc
        logging.info(f"Accuracy with noise '{name}': {acc:.4f}")

    return results


# --- [ Helper Functions - UNCHANGED ] ---
def occlude_points(data, num_occlusions=3):
    """(V18) Simulates occlusion by randomly setting keypoints to zero."""
    for i in range(data.shape[0]):
        occluded_indices = torch.randperm(data.shape[1])[:num_occlusions]
        data[i, occluded_indices, :] = 0
    return data


def structured_noise(data, noise_level=0.8):
    """(V18) Simulates structured noise (e.g., affecting one finger more)."""
    finger_indices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for i in range(data.shape[0]):
        finger_to_affect = finger_indices[torch.randint(0, 5, (1,)).item()]
        noise = torch.randn(len(finger_to_affect), data.shape[2], device=data.device) * noise_level
        if 0 <= min(finger_to_affect) and max(finger_to_affect) < data.shape[1]:
            data[i, finger_to_affect, :] += noise
    return data


def augment_spatial(inputs):
    """
    (V18 - Fixed) Applies rotation and scaling to (B, N, C) coordinates.
    Only augments the first 2 dimensions (x, y).
    """
    B, N, C = inputs.shape
    device = inputs.device

    xy_coords = inputs[..., :2]
    other_features = inputs[..., 2:]

    center = torch.mean(xy_coords, dim=1, keepdim=True)
    angles = (torch.rand(B, device=device) * 30 - 15) * (math.pi / 180.0)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rot_mats = torch.stack([torch.stack([cos_a, -sin_a], dim=1),
                            torch.stack([sin_a, cos_a], dim=1)], dim=1)
    xy_rotated = torch.einsum('bij,bkj->bki', rot_mats, xy_coords - center) + center

    scales = (torch.rand(B, 1, 1, device=device) * 0.2 + 0.9)
    xy_scaled = (xy_rotated - center) * scales + center

    return torch.cat([xy_scaled, other_features], dim=-1)


def plot_history(history, save_path):
    """(V18 - Unchanged) Plots training and validation loss/accuracy curves."""
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Training history plot saved to {save_path}")
    except Exception as e:
        logging.warning(f"Could not save training plot: {e}")
# --- [ Helper Functions END ] ---