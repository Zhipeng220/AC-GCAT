# config.py
import argparse


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate hand gesture recognition models (Refactored - 21 keypoints, Train/Test modes).")

    # --- 实验控制参数 ---
    parser.add_argument('--study_name', type=str, default='ablation',
                        # [MODIFIED] 添加 sota_only
                        choices=['ablation', 'full_ablation',
                                 'hyperparam_embed_dim', 'hyperparam_num_layers',
                                 'hyperparam_num_heads', 'hyperparam_cont_temp',
                                 'hyperparam_anat', 'hyperparam_contrastive',
                                 'sota_only'],  # <-- [新增]
                        help="Specify the study to run: 'ablation' for component analysis, "
                             "'hyperparam_embed_dim' for embedding dimension, "
                             "'hyperparam_num_layers' for attention layers, "
                             "'hyperparam_num_heads' for attention heads, "
                             "'hyperparam_cont_temp' for contrastive temperature, "
                             "'hyperparam_anat'/'hyperparam_contrastive' for loss weights, "
                             "'full_ablation' for the complete ablation study,"
                             "'sota_only' for running only the SOTA model.")  # <-- [新增]
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help="Set script to 'train' or 'test' mode.")

    # --- [ 新增: 随机数种子 ] ---
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (e.g., set to $i in a loop for stability tests).")
    # --- [ 新增结束 ] ---

    # ... (文件的其余部分保持不变) ...

    parser.add_argument('--train_csv_path', type=str,
                        # [修改] 指向您处理后的 CSV
                        default="/Users/gzp/Desktop/静态手势识别/data/ASL_Alphabet_Processed_Full_ResNet50/all_landmarks_multimodal.csv",
                        help="Path to the training CSV file (should include 'image_path' column).")
    parser.add_argument('--test_csv_path', type=str,
                        # [修改] 指向测试 CSV (如果 test 模式需要)
                        default="/Users/gzp/Desktop/静态手势识别/data/ASL_Alphabet_Processed_Full/test_landmarks_multimodal.csv",
                        help="Path to the external test CSV file for test mode.")
    parser.add_argument('--image_features_path', type=str,
                        # [修改] 指向您生成的 NPY 文件
                        default="/Users/gzp/Desktop/静态手势识别/data/ASL_Alphabet_Processed_Full_ResNet50/all_image_features_resnet50.npy",
                        help="Path to the pre-extracted .npy file for image features.")
    parser.add_argument('--model_path', type=str, help="Path to the pre-trained model .pth file for testing.")
    parser.add_argument('--model_name', type=str,
                        help="Name of the model to test (e.g., 'Hybrid_GCN_Attention(Gated)').")
    parser.add_argument('--output_dir', type=str, default="./results",
                        help="Directory to save all outputs (logs, weights, plots, scalers, etc.)")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Max learning rate for OneCycleLR.")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for AdamW.")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help="Max norm for gradient clipping (0 for no clipping).")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help="Number of epochs to wait for validation accuracy improvement before stopping (0 to disable).")
    parser.add_argument('--anatomical_weight_initial', type=float, default=0.0,
                        help="Initial weight for anatomical loss.")
    parser.add_argument('--anatomical_weight_final', type=float, default=0.02,
                        help="Default final weight for anatomical loss.")
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help="Weight for Hierarchical Contrastive Loss.")
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help="Temperature for Hierarchical Contrastive Loss.")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of attention blocks.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--embed_dim', type=int, default=256, help="Embedding dimension.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate.")

    args, unknown = parser.parse_known_args()
    return args

