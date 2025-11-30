# utils.py
import logging
import os
import time
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def setup_logger(log_dir="."):
    """Configures the logger to output to console and a file."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"experiment_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filepath)  # Corrected typo
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    logging.info(f"Log will be saved to: {log_filepath}")
    return logger


def get_feature_columns(df):
    """Identifies feature columns in the dataframe."""
    # Prioritize columns starting with x_, y_, z_
    coord_cols = [col for col in df.columns if str(col).lower().startswith(('x_', 'y_', 'z_'))]
    if coord_cols: return coord_cols

    # Fallback: dynamically find columns excluding known non-feature ones
    logging.warning("Falling back to positional feature selection (excluding label and image_path).")
    non_feature_cols = ['label', 'image_path']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    if feature_cols: return feature_cols

    logging.error("FATAL: Could not determine feature columns.")
    sys.exit(1)


def calculate_angle(p1, p2, p3):
    """
    Vectorized calculation of the angle (radians) at p2.
    p1, p2, p3 shape: (num_samples, 3)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle


def add_advanced_features(X_reshaped):
    """
    (V18 Feature Engineering)
    Adds bone vectors, joint angles, and fingertip distances to keypoint data.
    Input X_reshaped shape: (num_samples, 21, input_dim_original)
    """
    num_samples, num_keypoints, input_dim_original = X_reshaped.shape

    # Always use only the first 3 coordinates (x, y, z) for geometric features
    coords = X_reshaped[..., :3]

    # 1. Bone Vectors (3 features)
    parent_map = {
        0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7,
        9: 0, 10: 9, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14, 16: 15,
        17: 0, 18: 17, 19: 18, 20: 19
    }
    bone_vectors = np.zeros((num_samples, num_keypoints, 3))
    for i in range(1, num_keypoints):
        parent_idx = parent_map[i]
        bone_vectors[:, i, :] = coords[:, i, :] - coords[:, parent_idx, :]

    # 2. Joint Angles (1 feature)
    angle_features = np.zeros((num_samples, num_keypoints, 1))
    angle_triplets = [
        [1, 2, 3], [2, 3, 4],  # Thumb
        [5, 6, 7], [6, 7, 8],  # Index
        [9, 10, 11], [10, 11, 12],  # Middle
        [13, 14, 15], [14, 15, 16],  # Ring
        [17, 18, 19], [18, 19, 20]  # Pinky
    ]
    for p1_idx, p2_idx, p3_idx in angle_triplets:
        p1, p2, p3 = coords[:, p1_idx], coords[:, p2_idx], coords[:, p3_idx]
        angles = calculate_angle(p1, p2, p3)
        angle_features[:, p2_idx, 0] = angles  # Store angle at the middle joint

    # 3. Fingertip Distances (4 features)
    fingertip_dist_features = np.zeros((num_samples, num_keypoints, 4))
    p_thumb_tip = coords[:, 4]
    other_tip_indices = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky

    for i, tip_idx in enumerate(other_tip_indices):
        p_tip = coords[:, tip_idx]
        dist = np.linalg.norm(p_thumb_tip - p_tip, axis=1)
        # Broadcast this distance to all 21 keypoints
        fingertip_dist_features[..., i] = dist[:, np.newaxis]

    # 4. Concatenate all features
    X_engineered = np.concatenate([
        X_reshaped,  # Original (N, 21, C_orig)
        bone_vectors,  # (N, 21, 3)
        angle_features,  # (N, 21, 1)
        fingertip_dist_features  # (N, 21, 4)
    ], axis=2)

    logging.info(
        f"Advanced feature engineering applied: Input dimension increased from {input_dim_original} to {X_engineered.shape[2]}")
    return X_engineered


# [MODIFIED] 添加 use_advanced_features 标志 和 seed 参数
def load_and_preprocess_train_data(csv_path, image_features_path, use_advanced_features=True, seed=42):
    """
    (V19 - Scheme B)
    Loads landmarks from CSV, pre-extracted features from NPY, applies feature
    engineering to landmarks, scales landmarks, splits, and returns TensorDatasets.
    """
    logging.info(f"Loading landmarks from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
        if 'image_path' not in data.columns:
            logging.warning(f"CSV {csv_path} does not contain 'image_path' column needed for multimodal alignment.")
    except FileNotFoundError:
        logging.error(f"FATAL: Landmark CSV file not found: {csv_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading landmark CSV: {e}");
        sys.exit(1)

    # Load image features
    logging.info(f"Loading pre-extracted image features from {image_features_path}...")
    try:
        image_features_np = np.load(image_features_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Image features file not found: {image_features_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading image features NPY: {e}");
        sys.exit(1)

    # Validate alignment
    if len(data) != len(image_features_np):
        logging.error(f"FATAL: Mismatch! CSV has {len(data)} rows but NPY file has {len(image_features_np)} features.");
        sys.exit(1)
    logging.info(f"Successfully loaded {len(image_features_np)} image features.")

    # Feature Engineering for Landmarks (V18 logic)
    feature_cols = get_feature_columns(data)
    features, labels = data[feature_cols].values, data['label'].values

    num_feature_columns = features.shape[1]
    if num_feature_columns % 21 != 0:
        logging.error(f"FATAL: Landmark feature count ({num_feature_columns}) is not a multiple of 21.");
        sys.exit(1)
    input_dim_original = num_feature_columns // 21  # Original dim (e.g., 3)

    X_reshaped = features.reshape(-len(data), 21, input_dim_original)

    # --- [MODIFIED] 条件性特征工程 ---
    if use_advanced_features:
        X_engineered_reshaped = add_advanced_features(X_reshaped)  # Apply V18 feature engineering
    else:
        X_engineered_reshaped = X_reshaped  # 使用原始 (N, 21, 3) 特征
        logging.info("Skipping advanced feature engineering. Using basic (3D) keypoints.")
    # --- [MODIFIED END] ---

    input_dim_engineered = X_engineered_reshaped.shape[2]  # New keypoint dim (e.g., 3 or 11)
    X_engineered_flat = X_engineered_reshaped.reshape(-len(data), 21 * input_dim_engineered)

    # Encode labels and scale landmark features
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_engineered_flat)  # Only scale landmarks

    # Split all three arrays together
    # --- [ 修改：使用 seed ] ---
    X_train_val, X_test, img_feat_train_val, img_feat_test, y_train_val, y_test = train_test_split(
        X_scaled_flat, image_features_np, encoded_labels,
        test_size=0.2, random_state=seed, stratify=encoded_labels
    )
    X_train, X_val, img_feat_train, img_feat_val, y_train, y_val = train_test_split(
        X_train_val, img_feat_train_val, y_train_val,
        test_size=0.25, random_state=seed, stratify=y_train_val
    )
    # --- [ 修改结束 ] ---

    # Weighted Sampler for training set imbalance
    class_counts = np.bincount(y_train)
    weights = 1. / (class_counts + 1e-8)
    # Give more weight to extremely rare classes if any (optional)
    # weights[class_counts < 10] *= 10
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)

    # Create TensorDatasets with three tensors
    data_dict = {}
    for name, X_flat, img_feat, y in [
        ('train', X_train, img_feat_train, y_train),
        ('val', X_val, img_feat_val, y_val),
        ('test', X_test, img_feat_test, y_test)
    ]:
        # Reshape keypoint features back to (N, 21, C_engineered)
        X_kpt_tensor = torch.tensor(X_flat.reshape(-1, 21, input_dim_engineered), dtype=torch.float32)
        # Image features tensor (N, C_image)
        X_img_tensor = torch.tensor(img_feat, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        # Create dataset with three tensors
        data_dict[name] = TensorDataset(X_kpt_tensor, X_img_tensor, y_tensor)

    logging.info(
        f"Data split: {len(data_dict['train'])} train, {len(data_dict['val'])} validation, {len(data_dict['test'])} test samples.")
    # Return the engineered keypoint dimension
    return data_dict, scaler, label_encoder, sampler, input_dim_engineered


# [MODIFIED] 添加 use_advanced_features 标志
def load_and_preprocess_test_data(csv_path, image_features_path, scaler, label_encoder, use_advanced_features=True):
    """
    (V19 - Scheme B)
    Loads landmarks, pre-extracted features for an external test set.
    Uses pre-fitted scaler and label encoder.
    """
    logging.info(f"Loading external test landmarks from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Landmark CSV file not found: {csv_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading landmark CSV: {e}");
        sys.exit(1)

    # Load image features
    logging.info(f"Loading pre-extracted image features from {image_features_path}...")
    try:
        image_features_np = np.load(image_features_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Image features file not found: {image_features_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading image features NPY: {e}");
        sys.exit(1)

    # Validate alignment
    if len(data) != len(image_features_np):
        logging.error(f"FATAL: Mismatch! CSV has {len(data)} rows but NPY file has {len(image_features_np)} features.");
        sys.exit(1)

    # Feature Engineering for Landmarks (V18 logic)
    feature_cols = get_feature_columns(data)
    features, labels = data[feature_cols].values, data['label'].values

    num_feature_columns = features.shape[1]
    if num_feature_columns % 21 != 0:
        logging.error(f"FATAL: Landmark feature count ({num_feature_columns}) is not a multiple of 21.");
        sys.exit(1)
    input_dim_original = num_feature_columns // 21

    X_reshaped = features.reshape(-len(data), 21, input_dim_original)

    # --- [MODIFIED] 条件性特征工程 ---
    if use_advanced_features:
        X_engineered_reshaped = add_advanced_features(X_reshaped)
    else:
        X_engineered_reshaped = X_reshaped
        logging.info("Skipping advanced feature engineering for test set.")
    # --- [MODIFIED END] ---

    input_dim_engineered = X_engineered_reshaped.shape[2]
    X_engineered_flat = X_engineered_reshaped.reshape(-len(data), 21 * input_dim_engineered)

    # Scale landmarks and encode labels using pre-fitted tools
    try:
        X_scaled_flat = scaler.transform(X_engineered_flat)
        encoded_labels = label_encoder.transform(labels)
    except Exception as e:
        logging.error(f"Error transforming test data: {e}");
        sys.exit(1)

    # Create TensorDataset with three tensors
    X_kpt_tensor = torch.tensor(X_scaled_flat.reshape(-1, 21, input_dim_engineered), dtype=torch.float32)
    X_img_tensor = torch.tensor(image_features_np, dtype=torch.float32)
    y_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    return TensorDataset(X_kpt_tensor, X_img_tensor, y_tensor)