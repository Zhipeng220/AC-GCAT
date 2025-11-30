# main.py
import torch
import logging
import time
import os
import sys
import joblib
import numpy as np
import argparse
import random
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from config import get_args
from utils import setup_logger, load_and_preprocess_train_data, load_and_preprocess_test_data
from model import MLPBaseline, AdvancedAttentionModel, HybridAttentionModel, ImageOnlyBaseline
from losses import AnatomicalLoss, HierarchicalContrastiveLoss
from train import train_model, final_test_and_report, test_robustness, plot_history


# --- [MODIFIED BY AI] ---
def define_experiments(args):
    """
    [MODIFIED FOR HYPERPARAMETER SWEEPS]
    Defines experiment lists based on the 'study_name' argument.
    """

    experiments = {}
    keypoints_desc = "21 keypoints"  # 默认

    # --- SOTA 模型的基线配置 (来自 Exp 10) ---
    sota_baseline_config = {
        "model_class": HybridAttentionModel,  #
        "use_advanced_features": True,  #
        "use_cross_attn": True,  #
        "use_gated_fusion": True,  #
        "use_gcn_branch": False,  # (已确认)
        "use_anatomical_loss": True,  #
        "use_contrastive_loss": True,  #
        "embed_dim": args.embed_dim,  #
        "num_layers": args.num_layers,  #
        "num_heads": args.num_heads,  #
        "contrastive_temperature": args.contrastive_temperature,  #
    }

    # --- 根据 study_name 选择要运行的实验 ---

    # --- [ 新增: SOTA_ONLY ] ---
    if args.study_name == 'sota_only':
        study_title = "SOTA Model Stability Test"
        keypoints_desc = "SOTA Model Only (Exp 10)"
        experiments['Exp 10 (Full SOTA)'] = {**sota_baseline_config}
    # --- [ 新增结束 ] ---

    # 1. 嵌入维度实验 (来自 LaTeX a)
    elif args.study_name == 'hyperparam_embed_dim':
        study_title = "Hyperparameter Search: Embedding Dimension"
        # ... (后续代码不变) ...
        values_to_test = [128, 256, 384, 512]

        for dim in values_to_test:
            if dim % args.num_heads != 0:
                logging.warning(f"Skipping embed_dim={dim}; not divisible by num_heads={args.num_heads}")
                continue
            exp_name = f"EmbedDim_{dim}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "embed_dim": dim,  # 覆盖参数
            }

    # 2. 注意力层数实验 (来自 LaTeX b)
    elif args.study_name == 'hyperparam_num_layers':  # 您需要使用这个名字运行
        study_title = "Hyperparameter Search: Number of Layers"
        # ... (后续代码不变) ...
        values_to_test = [2, 4, 6, 8]

        for layers in values_to_test:
            exp_name = f"NumLayers_{layers}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "num_layers": layers,  # 覆盖参数
            }

    # 3. 注意力头数实验 (来自 LaTeX c)
    elif args.study_name == 'hyperparam_num_heads':  # 您需要使用这个名字运行
        study_title = "Hyperparameter Search: Number of Heads"
        # ... (后续代码不变) ...
        values_to_test = [4, 8, 16]

        for heads in values_to_test:
            if args.embed_dim % heads != 0:
                logging.warning(f"Skipping num_heads={heads}; embed_dim={args.embed_dim} not divisible by it.")
                continue
            exp_name = f"NumHeads_{heads}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "num_heads": heads,  # 覆盖参数
            }

    # 4. 对比损失温度实验 (来自 LaTeX d)
    elif args.study_name == 'hyperparam_cont_temp':  # 您需要使用这个名字运行
        study_title = "Hyperparameter Search: Contrastive Temperature"
        # ... (后续代码不变) ...
        values_to_test = [0.05, 0.07, 0.1, 0.2, 0.5]

        for temp in values_to_test:
            exp_name = f"ContTemp_{temp:.2f}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "contrastive_temperature": temp,  # 覆盖参数
            }

    # 5. 运行完整的15个实验 (来自 config.py 'full_ablation')
    elif args.study_name == 'full_ablation':
        study_title = "Full 15-Experiment Ablation Study"
        # ... (后续代码不变) ...
        keypoints_desc = "21 keypoints (Basic 3D / Advanced 11D)"

        # --- 10个线性实验 ---
        experiments['Exp 1 (MLP Basic)'] = {"model_class": MLPBaseline, "use_advanced_features": False,
                                            "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 2 (MLP Adv)'] = {"model_class": MLPBaseline, "use_advanced_features": True,
                                          "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 3 (Attn Adv)'] = {"model_class": AdvancedAttentionModel, "use_advanced_features": True,
                                           "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 4 (Img Only)'] = {"model_class": ImageOnlyBaseline, "use_advanced_features": True,
                                           "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 5 (Late Fusion)'] = {"model_class": HybridAttentionModel, "use_advanced_features": True,
                                              "use_cross_attn": False, "use_gated_fusion": False,
                                              "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 6 (CrossAttn no Gate)'] = {"model_class": HybridAttentionModel, "use_advanced_features": True,
                                                    "use_cross_attn": True, "use_gated_fusion": False,
                                                    "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 7 (SOTA Arch + CE)'] = {**sota_baseline_config, "use_anatomical_loss": False,
                                                 "use_contrastive_loss": False}
        experiments['Exp 8 (SOTA Arch + CE/Anat)'] = {**sota_baseline_config, "use_contrastive_loss": False}
        experiments['Exp 9 (SOTA Arch + CE/Cont)'] = {**sota_baseline_config, "use_anatomical_loss": False}
        experiments['Exp 10 (Full SOTA)'] = {**sota_baseline_config}

        # --- 5个新的组合实验 ---
        experiments['New-1 (Attn Basic + CE)'] = {"model_class": AdvancedAttentionModel, "use_advanced_features": False,
                                                  "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['New-3 (Hybrid Basic + CE)'] = {"model_class": HybridAttentionModel, "use_advanced_features": False,
                                                    "use_cross_attn": True, "use_gated_fusion": True,
                                                    "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['New-4 (Attn Basic + SOTA Loss)'] = {"model_class": AdvancedAttentionModel,
                                                         "use_advanced_features": False, "use_anatomical_loss": True,
                                                         "use_contrastive_loss": True}
        experiments['New-6 (Attn Adv + SOTA Loss)'] = {"model_class": AdvancedAttentionModel,
                                                       "use_advanced_features": True, "use_anatomical_loss": True,
                                                       "use_contrastive_loss": True}
        experiments['New-7 (Hybrid Basic + SOTA Loss)'] = {"model_class": HybridAttentionModel,
                                                           "use_advanced_features": False, "use_cross_attn": True,
                                                           "use_gated_fusion": True, "use_anatomical_loss": True,
                                                           "use_contrastive_loss": True}

    # 其他来自 config.py 的研究 (例如 anat_weight, contrastive_weight)
    elif args.study_name == 'hyperparam_anat':
        study_title = "Hyperparameter Search: Anatomical Loss Weight"
        # ... (后续代码不变) ...
        keypoints_desc = "SOTA Model, varying Anat. Weight"
        values_to_test = [0.0, 0.01, 0.02, 0.05, 0.1]
        for w in values_to_test:
            exp_name = f"AnatW_{w}"
            experiments[exp_name] = {**sota_baseline_config, "anatomical_weight_final": w}  # 覆盖

    elif args.study_name == 'hyperparam_contrastive':
        study_title = "Hyperparameter Search: Contrastive Loss Weight"
        # ... (后续代码不变) ...
        keypoints_desc = "SOTA Model, varying Cont. Weight"
        values_to_test = [0.0, 0.05, 0.1, 0.15, 0.2]
        for w in values_to_test:
            exp_name = f"ContW_{w}"
            experiments[exp_name] = {**sota_baseline_config, "contrastive_weight": w}  # 覆盖

    else:
        logging.warning(f"未知的 'study_name': {args.study_name}. 运行默认的 'full_ablation'.")
        args.study_name = 'full_ablation'
        return define_experiments(args)

    logging.info(f"--- 准备运行 {len(experiments)} 个独特的实验: {study_title} ---")
    return study_title, experiments, keypoints_desc


# ... (main 函数和文件的其余部分保持不变) ...

def main(args):
    """Main function to orchestrate training or testing."""
    output_dir = args.output_dir
    paths = {
        'base': output_dir,
        'logs': os.path.join(output_dir, 'logs'),
        'weights': os.path.join(output_dir, 'weights'),
        'plots': os.path.join(output_dir, 'plots'),
        'cm': os.path.join(output_dir, 'confusion_matrix')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    setup_logger(log_dir=paths['logs'])
    start_main_time = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.info(f"Using random seed: {args.seed}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"All outputs will be saved to: {output_dir}")

    if args.mode == 'train':
        study_title, experiments, keypoints_desc = define_experiments(args)
        logging.info(f"\nStarting Study: {study_title}")

        final_results, robustness_results = {}, {}
        scaler, label_encoder = None, None

        for name, config in experiments.items():
            logging.info(f"\n{'=' * 25} Running Experiment: {name} {'=' * 25}")

            safe_name = name.replace(' ', '_').replace('(', '_').replace(')', '').replace('=', '').replace(',',
                                                                                                           '').replace(
                '/', '_').replace('.', 'p')

            use_advanced_features = config.get("use_advanced_features", True)
            logging.info(f"Config: use_advanced_features = {use_advanced_features}")

            datasets, current_scaler, current_label_encoder, train_sampler, input_dim = load_and_preprocess_train_data(
                args.train_csv_path, args.image_features_path,
                use_advanced_features=use_advanced_features,
                seed=args.seed
            )

            if scaler is None: scaler = current_scaler
            if label_encoder is None: label_encoder = current_label_encoder

            pin_memory = device.type == 'cuda'
            train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=4, pin_memory=pin_memory)
            val_loader = DataLoader(datasets['val'], batch_size=args.batch_size * 2, shuffle=False, num_workers=4,
                                    pin_memory=pin_memory)
            test_loader = DataLoader(datasets['test'], batch_size=args.batch_size * 2, shuffle=False, num_workers=4,
                                     pin_memory=pin_memory)

            ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            anatomical_criterion = AnatomicalLoss(device=device)
            if hasattr(anatomical_criterion, 'label_classes_'):
                anatomical_criterion.label_classes_ = np.array(label_encoder.classes_)

            current_cont_temp = config.get("contrastive_temperature", args.contrastive_temperature)
            logging.info(f"Config: contrastive_temperature = {current_cont_temp}")
            contrastive_criterion = HierarchicalContrastiveLoss(temperature=current_cont_temp)

            model_class = config["model_class"]
            num_classes = len(label_encoder.classes_)

            current_embed_dim = config.get("embed_dim", args.embed_dim)
            current_num_layers = config.get("num_layers", args.num_layers)
            current_num_heads = config.get("num_heads", args.num_heads)
            logging.info(
                f"Config: embed_dim={current_embed_dim}, num_layers={current_num_layers}, num_heads={current_num_heads}")

            if 'resnet50' in args.image_features_path.lower():
                IMAGE_FEATURE_DIM = 2048
            else:
                IMAGE_FEATURE_DIM = 512

            model_kwargs = {"input_dim": input_dim, "num_classes": num_classes,
                            "num_layers": current_num_layers,
                            "num_heads": current_num_heads,
                            "embed_dim": current_embed_dim,
                            "dropout": args.dropout}

            if model_class == MLPBaseline:
                flat_kpt_dim = 21 * input_dim
                model = MLPBaseline(input_size=flat_kpt_dim, num_classes=num_classes)
            elif model_class == AdvancedAttentionModel:
                model = AdvancedAttentionModel(**model_kwargs)
            elif model_class == ImageOnlyBaseline:
                model = ImageOnlyBaseline(image_feature_dim=IMAGE_FEATURE_DIM, num_classes=num_classes,
                                          dropout=args.dropout)
            elif model_class == HybridAttentionModel:
                model = HybridAttentionModel(
                    **model_kwargs,
                    image_feature_dim=IMAGE_FEATURE_DIM,
                    use_gated_fusion=config.get("use_gated_fusion", False),
                    use_dynamic_graph=config.get("use_dynamic_graph", False),
                    use_cross_attn=config.get("use_cross_attn", True),
                    use_gcn_branch=config.get("use_gcn_branch", False)
                )

            logging.info(
                f"Instantiated model: {model.__class__.__name__} with keypoint_dim: {input_dim}, embed_dim: {current_embed_dim}")

            optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            total_steps = args.epochs * len(train_loader)
            if total_steps == 0:
                logging.warning("Train loader is empty! Skipping training.")
                continue
            scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=0.25)

            model_path = os.path.join(paths['weights'], f"best_model_{safe_name}_21kpts.pth")
            start_time = time.time()

            current_anat_weight_final = config.get("anatomical_weight_final", args.anatomical_weight_final)
            current_contrastive_weight = config.get("contrastive_weight", args.contrastive_weight)

            train_args = argparse.Namespace(**vars(args))
            train_args.anatomical_weight_final = current_anat_weight_final

            logging.info(
                f"Config: anat_weight_final={current_anat_weight_final}, contrastive_weight={current_contrastive_weight}")

            history = train_model(model, model_path, train_loader, val_loader, optimizer, scheduler, ce_criterion,
                                  anatomical_criterion if config.get("use_anatomical_loss") else None,
                                  contrastive_criterion if config.get("use_contrastive_loss") else None,
                                  train_args, device,
                                  contrastive_weight=current_contrastive_weight)

            plot_path = os.path.join(paths['plots'], f"history_{safe_name}.png")
            plot_history(history, plot_path)

            test_acc = final_test_and_report(model_class, model_path, test_loader, device, label_encoder, args,
                                             input_dim, config, output_paths={'cm_dir': paths['cm']},
                                             image_feature_dim=IMAGE_FEATURE_DIM)
            final_results[name] = {'accuracy': test_acc, 'train_time': time.time() - start_time}

            if use_advanced_features:
                robust_res = test_robustness(model_class, model_path, datasets['test'], device, label_encoder, args,
                                             input_dim, config, image_feature_dim=IMAGE_FEATURE_DIM)
                if robust_res: robustness_results[name] = robust_res
            else:
                logging.info("Skipping robustness test for basic features experiment.")

        if scaler and label_encoder:
            joblib.dump(scaler, os.path.join(output_dir, 'scaler_21kpts.pkl'))
            joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder_21kpts.pkl'))
            logging.info(f"\nPreprocessing tools saved to: {output_dir}")

        summary_path = os.path.join(output_dir,
                                    f"summary_report_{args.study_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"{'=' * (len(study_title) + 4)}\n")
            f.write(f"  {study_title} Summary ({keypoints_desc})\n")
            f.write(f"{'=' * (len(study_title) + 4)}\n\n")
            header = "Experiment Name".ljust(40) + "| Accuracy      | Robustness (gaussian_1.0) | Train Time (s) "
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            sorted_results = sorted(final_results.items(), key=lambda item: item[1]['accuracy'], reverse=True)

            for name, results in sorted_results:
                acc_str = f"{results['accuracy'] * 100:.2f}%".ljust(13)
                time_str = f"{results['train_time']:.2f}s".ljust(14)
                robust_str = "N/A".ljust(25)
                if name in robustness_results and 'gaussian_1.0' in robustness_results[name]:
                    robust_str = f"{robustness_results[name]['gaussian_1.0'] * 100:.2f}%".ljust(25)
                f.write(f"{name.ljust(40)}| {acc_str} | {robust_str} | {time_str}\n")
            f.write("=" * len(header) + "\n")
        logging.info(f"Summary report saved to {summary_path}")

    elif args.mode == 'test':
        if not all([args.test_csv_path, args.image_features_path, args.model_path, args.model_name, args.output_dir]):
            logging.error(
                "FATAL: For 'test' mode, --test_csv_path, --image_features_path, --model_path, --model_name, and --output_dir are required.");
            sys.exit(1)

        try:
            scaler_path = os.path.join(args.output_dir, 'scaler_21kpts.pkl')
            encoder_path = os.path.join(args.output_dir, 'label_encoder_21kpts.pkl')
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
            logging.info("Loaded pre-fitted scaler and label encoder.")
        except FileNotFoundError:
            logging.error(
                f"FATAL: scaler ({scaler_path}) or label_encoder ({encoder_path}) not found in {args.output_dir}. Run in 'train' mode first.");
            sys.exit(1)

        logging.warning("Assuming 'test' mode uses ADVANCED features.")
        test_dataset = load_and_preprocess_test_data(
            args.test_csv_path, args.image_features_path, scaler, label_encoder,
            use_advanced_features=True
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

        model_mapping = {
            'MLPBaseline': (MLPBaseline, {}),
            'AdvancedAttentionModel': (AdvancedAttentionModel, {}),
            'HybridAttentionModel': (HybridAttentionModel, {
                'use_gated_fusion': True, 'use_dynamic_graph': False,
                'use_cross_attn': True, 'use_gcn_branch': False
            }),
            'ImageOnlyBaseline': (ImageOnlyBaseline, {})
        }
        if args.model_name not in model_mapping:
            logging.error(f"FATAL: Unknown model name '{args.model_name}'. Choose from {list(model_mapping.keys())}");
            sys.exit(1)
        model_class, config = model_mapping[args.model_name]

        cm_dir = os.path.join(args.output_dir, 'confusion_matrix')
        os.makedirs(cm_dir, exist_ok=True)
        input_dim = test_dataset.tensors[0].shape[-1]

        if 'resnet50' in args.image_features_path.lower():
            IMAGE_FEATURE_DIM = 2048
        else:
            IMAGE_FEATURE_DIM = 512
        logging.info(f"Detected image feature dimension for test: {IMAGE_FEATURE_DIM}")

        final_test_and_report(model_class, args.model_path, test_loader, device, label_encoder, args, input_dim, config,
                              output_paths={'cm_dir': cm_dir}, image_feature_dim=IMAGE_FEATURE_DIM)

    logging.info(f"\nTotal execution time: {(time.time() - start_main_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    args = get_args()
    logging.info(f"Running study: {args.study_name}")
    main(args)

    # !/bin/bash
    # run_all_datasets.sh

    # 设置为在第一个错误处停止
    """
    set - e

    echo
    "--- [ 1/4 ] 正在运行: NUS Hand Posture Dataset ---"
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/NUS_Hand_Posture/all_landmarks_multimodal.csv" \
    - -image_features_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/NUS_Hand_Posture/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_NUS_Hand_Posture"

    echo
    "--- [ 2/4 ] 正在运行: Indian Sign Language Dataset ---"
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/Indian Sign Language/all_landmarks_multimodal.csv" \
    - -image_features_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/Indian Sign Language/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_Indian_Sign_Language"

    echo
    "--- [ 3/4 ] 正在运行: asl_dataset (新) ---"
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/asl_dataset/all_landmarks_multimodal.csv" \
    - -image_features_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/asl_dataset/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_asl_dataset"

    echo
    "--- [ 4/4 ] 正在运行: ASL Alphabet (旧) ---"
    # 注意：这些路径来自你 config.py 的默认值
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/ASL_Alphabet/all_landmarks_multimodal.csv" \
    - -image_features_path
    "/Users/gzp/Desktop/AC-GCAT/AC-GCAT/data/ASL_Alphabet/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_ASL_Alphabet_Old"

    echo
    "--- 所有 4 个数据集的实验均已完成 ---"
    """
