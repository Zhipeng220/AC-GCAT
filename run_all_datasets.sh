#!/bin/bash
# run_all_datasets.sh

# 设置为在第一个错误处停止
set -e

echo "--- [ 1/4 ] 正在运行: NUS Hand Posture Dataset ---"
python main.py --study_name sota_only --epochs 50 \
    --train_csv_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/NUS_Hand_Posture/all_landmarks_multimodal.csv" \
    --image_features_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/NUS_Hand_Posture/all_image_features_resnet50.npy" \
    --output_dir "./results_NUS_Hand_Posture"

echo "--- [ 2/4 ] 正在运行: Indian Sign Language Dataset ---"
python main.py --study_name sota_only --epochs 50 \
    --train_csv_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/Indian Sign Language/all_landmarks_multimodal.csv" \
    --image_features_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/Indian Sign Language/all_image_features_resnet50.npy" \
    --output_dir "./results_Indian_Sign_Language"

echo "--- [ 3/4 ] 正在运行: asl_dataset (新) ---"
python main.py --study_name sota_only --epochs 50 \
    --train_csv_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/asl_dataset/all_landmarks_multimodal.csv" \
    --image_features_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/asl_dataset/all_image_features_resnet50.npy" \
    --output_dir "./results_asl_dataset"

echo "--- [ 4/4 ] 正在运行: ASL Alphabet (旧) ---"
python main.py --study_name sota_only --epochs 50 \
    --train_csv_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/ASL_Alphabet/all_landmarks_multimodal.csv" \
    --image_features_path "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/ASL_Alphabet/all_image_features_resnet50.npy" \
    --output_dir "./results_ASL_Alphabet_Old"

echo "--- 所有 4 个数据集的实验均已完成 ---"