from __future__ import annotations

from typing import TYPE_CHECKING

import os
from argparse import ArgumentParser

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .data import EDCDataModule
from .model import EDCModel, EDCPredsWriter

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = [
    "build_trainer",
    "build_modules",
    "build_cli_arg_parser"
]

AMP_PRECISIONS = {
    "fp16": 16,
    "bf16": "bf16",
    "fp32": 32
}

GRAD_COMM_HOOKS = {
    "fp16": ddp_hooks.fp16_compress_hook,
    "bf16": ddp_hooks.bf16_compress_hook,
    "fp32": None
}

def build_trainer(args: Namespace, mode: str) -> Trainer:
    strategy_name = args.strategy

    # Use prediction writer callback for prediction mode
    if mode=="train":
        # Enable checkpointing
        callbacks = [ModelCheckpoint(
            filename="model",
            monitor="val_accuracy",
            save_weights_only=True,
            mode="max"
        )]
        # Logging
        logger = TensorBoardLogger(save_dir="logs", name=args.name)
    # Prediction mode
    elif mode=="predict":
        # Collect and write predictions
        callbacks = [EDCPredsWriter(
            outputs_path=args.outputs_path,
            standalone_ctx=getattr(args, "standalone_ctx", False)
        )]
        # Supress logging
        logger = False
    # Unknown mode
    else:
        raise ValueError(f"unknown mode: {mode}")

    # DDP training strategy
    if strategy_name=="ddp":
        grad_comm_dtype = getattr(args, "grad_comm_dtype", None)

        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            static_graph=True,
            ddp_comm_hook=GRAD_COMM_HOOKS.get(grad_comm_dtype)
        )
    # Other strategies
    else:
        strategy = strategy_name
    
    return Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=getattr(args, "epochs", None),
        accelerator=args.device,
        strategy=strategy,
        precision=AMP_PRECISIONS[args.amp_dtype],
        gradient_clip_val=getattr(args, "grad_clip_val", None),
        val_check_interval=0.2
    )

def build_modules(args: Namespace, mode: str) -> tuple[EDCModel, EDCDataModule]:
    # Build data module
    data_module = EDCDataModule(
        dataset_path=args.dataset,
        transformer_name=args.transformer_name,
        max_rounds=args.max_rounds,
        max_ctx_len=args.max_ctx_len,
        with_user_action=not args.no_user_action,
        with_sys_action=not args.no_sys_action,
        one_pass=args.one_pass,
        standalone_ctx=args.standalone_ctx,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        predict_subset=getattr(args, "subset", "dev")
    )
    
    if mode=="predict":
        # Get checkpoint path
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        checkpoint_path = os.path.join(logs_dir, args.name, "checkpoints", "model.ckpt")
        # Load model from existing checkpoint
        model = EDCModel.load_from_checkpoint(checkpoint_path)
    else:
        # Create model based 
        model = EDCModel(
            max_rounds=args.max_rounds,
            transformer_name=args.transformer_name,
            learning_rate=args.learning_rate,
            skip_init_model=False
        )

    return model, data_module

def build_cli_arg_parser(mode: str) -> ArgumentParser:
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument(
        "-n", "--name", required=True, help="name of the experiment"
    )
    parser.add_argument(
        "--strategy", choices=["ddp", "ddp_sharded"], default="ddp",
        help="training or prediction strategy"
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="gpu",
        help="device for training or prediction"
    )
    parser.add_argument(
        "--amp-dtype", choices=list(AMP_PRECISIONS.keys()), default="fp16",
        help="data type for mixed precision training or prediction"
    )

    # Data and model arguments
    parser.add_argument("-d", "--dataset", required=True, help="path to JSON dataset file")
    parser.add_argument("-b", "--batch-size", type=int, help="batch size")
    parser.add_argument(
        "--max-rounds", type=int, default=24,
        help="maximum number of dialog rounds included as context"
    )
    parser.add_argument(
        "--max-ctx-len", type=int, default=1024, help="maximum context sequence length"
    )
    parser.add_argument(
        "--no-user-action", default=False, action="store_true",
        help="do not include user action prediction auxiliary task"
    )
    parser.add_argument(
        "--no-sys-action", default=False, action="store_true",
        help="do not include system action prediction auxiliary task"
    )
    parser.add_argument(
        "--one-pass", default=False, action="store_true",
        help="predict dialog state and actions in one pass"
    )
    parser.add_argument(
        "--standalone-ctx", default=False, action="store_true",
        help="use standalone instead of shared context encoding"
    )
    parser.add_argument(
        "--n-workers", type=int, default=2,
        help="number of worker processes for data loading"
    )
    parser.add_argument(
        "--transformer-name", default="facebook/bart-base",
        help="name of underlying Transformer model"
    )

    if mode=="train":
        # Trainer arguments
        parser.add_argument(
            "-e", "--epochs", type=int, default=5, help="number of training epochs"
        )
        parser.add_argument(
            "-l", "--learning-rate", type=float, default=1.5e-4,
            help="maximum learning rate for training"
        )
        parser.add_argument(
            "-s", "--seed", type=int,
            help="random seed for reproducible training results"
        )
        parser.add_argument(
            "--grad-comm-dtype", choices=list(GRAD_COMM_HOOKS.keys()), default="fp16",
            help="data type for compressed gradient communication during training"
        )
        parser.add_argument(
            "--grad-clip-val", type=float, default=1., help="value to clip gradient to"
        )
    elif mode=="predict":
        # Trainer (callback) arguments
        parser.add_argument(
            "-o", "--outputs-path", default="preds.zip",
            help="path to save prediction results"
        )
        parser.add_argument(
            "-s", "--subset", default="dev", help="subset to run prediction on"
        )
    else:
        raise ValueError(f"unknown CLI mode: '{mode}'")

    return parser
