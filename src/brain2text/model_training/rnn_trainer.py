import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle
from brain2text.lm.local_lm import LocalNgramDecoder, LocalLMConfig
try:
    import wandb
except ImportError:
    wandb = None

import editdistance
from .evaluate_model_helpers import remove_punctuation

from .dataset import BrainToTextDataset, train_test_split_indicies
from .data_augmentations import gauss_smooth


from omegaconf import OmegaConf

def _normalize_for_wer(s: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z\- \']", "", s)
    s = s.replace("- ", " ").lower()
    s = s.replace("--", "").lower()
    s = s.replace(" '", "'").lower()
    s = s.strip()
    s = " ".join([w for w in s.split() if w])
    return s


def _decode_transcription_to_str(x) -> str:
    """
    Decode an HDF5 transcription field into a Python string.
    The dataset stores g['transcription'] as a numeric array (typically uint8 ASCII).
    After torch.stack it becomes a torch.Tensor of bytes. Convert to bytes->utf-8.
    """
    import numpy as np
    import torch

    if isinstance(x, str):
        return x

    if isinstance(x, (bytes, np.bytes_)):
        b = bytes(x)
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy().reshape(-1)
        b = arr.astype(np.uint8, copy=False).tobytes()
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    if isinstance(x, np.ndarray):
        arr = np.asarray(x).reshape(-1)
        b = arr.astype(np.uint8, copy=False).tobytes()
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    return str(x)



torch.set_float32_matmul_precision('high') # makes float32 matmuls faster on some GPUs
torch.backends.cudnn.deterministic = True # makes training more reproducible
# --- opcional: solo Torch >= 2 ---
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

torch.backends.cudnn.deterministic = True

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.cache_size_limit = 64

from .rnn_model import GRUDecoder, ResLSTMDecoder, XLSTMDecoder
class BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder
    
    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing
        self.best_val_WER = float("inf")  # track best WER for checkpointing (lower is better)


        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args["output_dir"], exist_ok=True)


        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use 
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Optional: Weights & Biases logging
        self.use_wandb = bool(self.args.get("wandb", {}).get("enabled", False))
        wandb_group = self.args["wandb"].get("group", None) or os.environ.get("WANDB_RUN_GROUP")
        wandb_job_type = self.args["wandb"].get("job_type", None) or os.environ.get("WANDB_JOB_TYPE")

        if self.use_wandb:
            if wandb is None:
                raise ImportError("wandb is enabled in rnn_args.yaml but wandb is not installed. Run: pip install wandb")
            wandb.init(
                project=self.args["wandb"]["project"],
                name=self.args["wandb"].get("run_name", None),
                tags=self.args["wandb"].get("tags", None),
                config=OmegaConf.to_container(self.args, resolve=True),
                group=wandb_group,
                job_type=wandb_job_type,
            )

       # ---------------- WER (local LM) ----------------
        self.eval_cfg = self.args.get("eval", {})
        self.compute_wer = bool(self.eval_cfg.get("compute_wer", False))
        self._val_step_count = 0
        self._lm = None

        from pathlib import Path

        if self.compute_wer:
            try:
                repo_root = Path(__file__).resolve().parents[3]
                lm_dir_path = Path(str(self.eval_cfg["lm_dir"]))
                if not lm_dir_path.is_absolute():
                    lm_dir_path = repo_root / lm_dir_path
                cfg = LocalLMConfig(
                    lm_dir=str(lm_dir_path),
                    max_active=int(self.eval_cfg.get("max_active", 7000)),
                    min_active=int(self.eval_cfg.get("min_active", 200)),
                    beam=float(self.eval_cfg.get("beam", 15.0)),
                    lattice_beam=float(self.eval_cfg.get("lattice_beam", 8.0)),
                    ctc_blank_skip_threshold=float(self.eval_cfg.get("ctc_blank_skip_threshold", 0.95)),
                    length_penalty=float(self.eval_cfg.get("length_penalty", 0.0)),
                    acoustic_scale=float(self.eval_cfg.get("acoustic_scale", 0.35)),
                    nbest=int(self.eval_cfg.get("nbest", 50)),
                    blank_penalty=float(self.eval_cfg.get("blank_penalty", 90.0)),
                    reorder_mode=str(self.eval_cfg.get("reorder_mode", "identity")),
                    sil_index=int(self.eval_cfg.get("sil_index", -1)),
                )
                self._lm = LocalNgramDecoder(cfg)

                # --- Optional 5-gram decoder (safe: does not affect existing 1-gram path) ---
                self._lm_5gram = None
                lm_dir_5gram = self.eval_cfg.get("lm_dir_5gram", None)

                if lm_dir_5gram:
                    try:
                        lm_dir_5gram_path = Path(str(lm_dir_5gram))
                        if not lm_dir_5gram_path.is_absolute():
                            lm_dir_5gram_path = repo_root / lm_dir_5gram_path

                        cfg5 = LocalLMConfig(
                            lm_dir=str(lm_dir_5gram_path),
                            max_active=cfg.max_active,
                            min_active=cfg.min_active,
                            beam=cfg.beam,
                            lattice_beam=cfg.lattice_beam,
                            ctc_blank_skip_threshold=cfg.ctc_blank_skip_threshold,
                            length_penalty=cfg.length_penalty,
                            acoustic_scale=cfg.acoustic_scale,
                            nbest=cfg.nbest,
                            blank_penalty=cfg.blank_penalty,
                            reorder_mode=cfg.reorder_mode,
                            sil_index=cfg.sil_index,
                        )
                        self._lm_5gram = LocalNgramDecoder(cfg5)
                        self.logger.info(f"Local LM 5-gram enabled. lm_dir_5gram={cfg5.lm_dir}")
                    except Exception as e:
                        self.logger.warning(f"Could not init 5-gram LM decoder from eval.lm_dir_5gram={lm_dir_5gram!r}: {e}")
                        self._lm_5gram = None


                self.logger.info(f"Local LM WER enabled. lm_dir={cfg.lm_dir}")
            except Exception as e:
                self.logger.warning(
                    "Local LM WER requested (eval.compute_wer=true) but could not initialize lm_decoder. "
                    f"Disabling WER for this run and continuing. Reason: {e}"
                )
                self.compute_wer = False
                self._lm = None
                self._lm_5gram = None

        # ------------------------------------------------


        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])


        # Filter out sessions that have no training trials (missing/empty data_train.hdf5)
        try:
            import h5py
        except ImportError:
            h5py = None

        filtered_sessions = []
        filtered_val_probs = []

        dataset_dir = self.args["dataset"]["dataset_dir"]
        sessions = list(self.args["dataset"]["sessions"])
        val_probs = list(self.args["dataset"]["dataset_probability_val"])

        for s, p in zip(sessions, val_probs):
            train_fp = os.path.join(dataset_dir, s, "data_train.hdf5")
            if not os.path.exists(train_fp):
                self.logger.warning(f"Skipping session {s}: missing {train_fp}")
                continue
            if h5py is not None:
                try:
                    with h5py.File(train_fp, "r") as f:
                        if len(f.keys()) == 0:
                            self.logger.warning(f"Skipping session {s}: train file has 0 trials ({train_fp})")
                            continue
                except Exception as e:
                    self.logger.warning(f"Skipping session {s}: could not read {train_fp} ({e})")
                    continue

            filtered_sessions.append(s)
            filtered_val_probs.append(p)

        if len(filtered_sessions) == 0:
            raise RuntimeError("No valid training sessions found. Check dataset_dir and data files.")

        self.args["dataset"]["sessions"] = filtered_sessions
        self.args["dataset"]["dataset_probability_val"] = filtered_val_probs
        self.logger.info(f"Using {len(filtered_sessions)} sessions after filtering (from {len(sessions)}).")



        # Initialize the model (selectable via config)
        mcfg = self.args.get("model", {})
        decoder_type = str(mcfg.get("decoder_type", "gru")).lower()

        if decoder_type == "gru":
            DecoderCls = GRUDecoder
        elif decoder_type == "reslstm":
            DecoderCls = ResLSTMDecoder
        elif decoder_type == "xlstm":
            DecoderCls = XLSTMDecoder
        else:
            raise ValueError(f"Invalid model.decoder_type: {decoder_type}. Use 'gru', 'reslstm', or 'xlstm'.")

        decoder_kwargs = dict(
            neural_dim=mcfg["n_input_features"],
            n_units=mcfg["n_units"],
            n_days=len(self.args["dataset"]["sessions"]),
            n_classes=self.args["dataset"]["n_classes"],
            rnn_dropout=mcfg["rnn_dropout"],
            input_dropout=mcfg["input_network"]["input_layer_dropout"],
            n_layers=mcfg["n_layers"],
            patch_size=mcfg["patch_size"],
            patch_stride=mcfg["patch_stride"],
        )

        # Extra args only for ResLSTMDecoder (safe defaults if not present)
        if DecoderCls is ResLSTMDecoder:
            decoder_kwargs.update(dict(
                reslstm_num_blocks=int(mcfg.get("reslstm_num_blocks", 1)),
                reslstm_sublayers_per_block=int(mcfg.get("reslstm_sublayers_per_block", 2)),
                reslstm_lstm_layers=int(mcfg.get("reslstm_lstm_layers", 2)),
                reslstm_lstm_dropout=float(mcfg.get("reslstm_lstm_dropout", 0.1)),
                reslstm_norm=str(mcfg.get("reslstm_norm", "bn")),
                reslstm_pre_norm=bool(mcfg.get("reslstm_pre_norm", False)),
                reslstm_residual_dropout=float(mcfg.get("reslstm_residual_dropout", 0.0)),
            ))

        # Extra args only for XLSTMDecoder
        if DecoderCls is XLSTMDecoder:
            decoder_kwargs.update(dict(
                xlstm_num_blocks=int(mcfg.get("xlstm_num_blocks", mcfg["n_layers"])),
                xlstm_num_heads=int(mcfg.get("xlstm_num_heads", 4)),
                xlstm_conv1d_kernel_size=int(mcfg.get("xlstm_conv1d_kernel_size", 4)),
                xlstm_dropout=float(mcfg.get("xlstm_dropout", mcfg["rnn_dropout"])),

                # Match GRU feature set (head + speckle)
                head_type=str(mcfg.get("head_type", "none")),
                head_num_blocks=int(mcfg.get("head_num_blocks", 0)),
                head_norm=str(mcfg.get("head_norm", "none")),
                head_dropout=float(mcfg.get("head_dropout", 0.0)),
                head_activation=str(mcfg.get("head_activation", "gelu")),
                input_speckle_p=float(mcfg.get("input_speckle_p", 0.0)),
                input_speckle_mode=str(mcfg.get("input_speckle_mode", "feature")),
            ))


        # GRU-only knobs (ONLY keep this block if your GRUDecoder __init__ supports these kwargs)
        if DecoderCls is GRUDecoder:
            decoder_kwargs.update(dict(
                head_type=str(mcfg.get("head_type", "none")),
                head_num_blocks=int(mcfg.get("head_num_blocks", 0)),
                head_norm=str(mcfg.get("head_norm", "none")),
                head_dropout=float(mcfg.get("head_dropout", 0.0)),
                head_activation=str(mcfg.get("head_activation", "gelu")),
                input_speckle_p=float(mcfg.get("input_speckle_p", 0.0)),
                input_speckle_mode=str(mcfg.get("input_speckle_mode", "feature")),
            ))

        # Make it robust: drop any kwargs not accepted by the selected decoder
        import inspect
        allowed = set(inspect.signature(DecoderCls.__init__).parameters.keys())
        allowed.discard("self")
        decoder_kwargs = {k: v for k, v in decoder_kwargs.items() if k in allowed}

        self.model = DecoderCls(**decoder_kwargs)



        # Call torch.compile to speed up training
        self.logger.info("Using torch.compile (if available)")
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("torch.compile enabled.")
            except Exception as e:
                self.logger.warning(f"torch.compile failed; falling back to eager. Reason: {e}")
        else:
            self.logger.info("torch.compile not available (torch<2.0). Skipping.")


        self.logger.info(f"Initialized RNN decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Create datasets and dataloaders
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

        # Ensure that there are no duplicate days
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        # Split trials into train and test sets
        train_trials, _ = train_test_split_indicies(
            file_paths = train_file_paths, 
            test_percentage = 0,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        _, val_trials = train_test_split_indicies(
            file_paths = val_file_paths, 
            test_percentage = 1,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        
        # --- ensure output/checkpoint dirs exist ---
        self.args["output_dir"] = str(self.args["output_dir"])
        if "checkpoint_dir" in self.args and self.args["checkpoint_dir"] is not None:
            self.args["checkpoint_dir"] = str(self.args["checkpoint_dir"])
        else:
            self.args["checkpoint_dir"] = os.path.join(self.args["output_dir"], "checkpoint")

        os.makedirs(self.args["output_dir"], exist_ok=True)
        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)
        # ------------------------------------------


        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')
            
        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True 
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, 
            num_workers = 0,
            pin_memory = True 
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=self.args['lr_decay_steps'],
            )

        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)

        elif self.args['lr_scheduler_type'] == 'cosine_stepdrop':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer, use_stepdrop=True)

        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")

        
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Set rnn and/or input layers to not trainable if specified 
        for name, param in self.model.named_parameters():
            if (not self.args["model"]["rnn_trainable"]) and (("gru" in name) or ("lstm" in name) or ("xlstm" in name)):
                param.requires_grad = False

            elif (not self.args["model"]["input_network"]["input_trainable"]) and ("day_" in name):
                param.requires_grad = False


        # Send model to device 
        self.model.to(self.device)

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        day_params = [p for name, p in self.model.named_parameters() if "day_" in name]

        # No weight decay for biases and normalization params (LayerNorm etc.), excluding day_ (day_ has its own group)
        no_decay_params = [
            p for name, p in self.model.named_parameters()
            if ("day_" not in name) and (("bias" in name) or ("norm" in name) or ("bn" in name))
        ]


        other_params = [
            p for name, p in self.model.named_parameters()
            if ("day_" not in name) and ("bias" not in name) and ("norm" not in name) and ("bn" not in name)
        ]



        if len(day_params) != 0:
            param_groups = [
                    {'params' : no_decay_params, 'weight_decay' : 0, 'group_type' : 'no_decay'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : no_decay_params, 'weight_decay' : 0, 'group_type' : 'no_decay'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
            
        # AdamW: fused solo si está disponible en tu torch
        adamw_kwargs = dict(
            lr=self.args["lr_max"],
            betas=(self.args["beta0"], self.args["beta1"]),
            eps=self.args["epsilon"],
            weight_decay=self.args["weight_decay"],
        )

        try:
            optim = torch.optim.AdamW(
                param_groups,
                fused=True,
                **adamw_kwargs,
            )
            self.logger.info("AdamW(fused=True) enabled.")
        except TypeError:
            optim = torch.optim.AdamW(
                param_groups,
                **adamw_kwargs,
            )
            self.logger.info("AdamW(fused) not available in this torch build. Using standard AdamW.")

        return optim 

    def create_cosine_lr_scheduler(self, optim, use_stepdrop: bool = False):

        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        # Optional step-drop params (only used if use_stepdrop=True)
        stepdrop_step = int(self.args.get("lr_stepdrop_step", -1))
        stepdrop_factor = float(self.args.get("lr_stepdrop_factor", 1.0))
        if use_stepdrop and (stepdrop_step < 0 or stepdrop_factor <= 0):
            raise ValueError(f"Invalid stepdrop config: lr_stepdrop_step={stepdrop_step}, lr_stepdrop_factor={stepdrop_factor}")


        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            # Warmup
            if current_step < warmup_steps:
                base = float(current_step + 1) / float(max(1, warmup_steps))
            # Cosine decay
            elif current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                base = max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            else:
                base = min_lr_ratio

            # Apply stepdrop (multiplicative) if enabled
            if use_stepdrop and current_step >= stepdrop_step:
                base = base * stepdrop_factor

            return base


        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        ''' 
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only = False) # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf
        self.best_val_WER = checkpoint.get('val_WER', float("inf"))


        self.model.to(self.device)
        
        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss, WER=None):

        '''
        Save a training checkpoint
        '''

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_PER': PER,
            'val_loss': loss,
            'val_WER': WER,
        }

        
        torch.save(checkpoint, save_path)
        
        self.logger.info("Saved model to checkpoint: " + save_path)

        # Save the args file alongside the checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def create_attention_mask(self, sequence_lengths):

        max_length = torch.max(sequence_lengths).item()

        batch_size = sequence_lengths.size(0)
        
        # Create a mask for valid key positions (columns)
        # Shape: [batch_size, max_length]
        key_mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length)
        key_mask = key_mask < sequence_lengths.unsqueeze(1)
        
        # Expand key_mask to [batch_size, 1, 1, max_length]
        # This will be broadcast across all query positions
        key_mask = key_mask.unsqueeze(1).unsqueeze(1)
        
        # Create the attention mask of shape [batch_size, 1, max_length, max_length]
        # by broadcasting key_mask across all query positions
        attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)
        
        # Convert boolean mask to float mask:
        # - True (valid key positions) -> 0.0 (no change to attention scores)
        # - False (padding key positions) -> -inf (will become 0 after softmax)
        attention_mask_float = torch.where(attention_mask, 
                                        True,
                                        False)
        
        return attention_mask_float

    def transform_data(self, features, n_time_steps, mode = 'train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        
        return features, n_time_steps

    def train(self):
        '''
        Train the model 
        '''

        # Set model to train mode (specificially to make sure dropout layers are engaged)
        self.model.train()

        # create vars to track performance
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        # training params 
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)

        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()


        # train for specified number of batches
        for i, batch in enumerate(self.train_loader):
            
            self.model.train()
            self.optimizer.zero_grad()
            
            # Train step
            start_time = time.time() 

            # Move data to device
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Use autocast for efficiency
            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):

                # Apply augmentations to the data
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                ps = int(self.args["model"]["patch_size"])
                st = int(self.args["model"]["patch_stride"])

                if ps > 0:
                    if st <= 0:
                        raise ValueError(f"Invalid patch_stride={st} with patch_size={ps}")
                    adjusted_lens = torch.div((n_time_steps - ps), st, rounding_mode="floor") + 1
                    adjusted_lens = adjusted_lens.to(torch.int32)

                else:
                    adjusted_lens = n_time_steps.to(torch.int32)


                # Get phoneme predictions 
                logits = self.model(features, day_indicies)

                # Calculate CTC Loss
                loss = self.ctc_loss(
                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets = labels,
                    input_lengths = adjusted_lens,
                    target_lengths = phone_seq_lens
                    )
                    
                loss = torch.mean(loss) # take mean loss over batches
            
                loss.backward()

                # Skip step if gradients are non-finite (NaN/Inf)
                if self.args["grad_norm_clip_value"] > 0:
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.args["grad_norm_clip_value"],
                            error_if_nonfinite=False,
                            foreach=True,   # solo si existe
                        )
                    except TypeError:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.args["grad_norm_clip_value"],
                            error_if_nonfinite=False,
                        )
                else:
                    grad_norm = torch.tensor(float("nan"), device=self.device)

                if not torch.isfinite(grad_norm):
                    self.logger.warning(f"Non-finite grad norm at step {i}: {grad_norm}. Skipping optimizer step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    # optional: also skip scheduler step to keep LR schedule consistent
                    continue

                used_lrs = [pg["lr"] for pg in self.optimizer.param_groups]


                # Advance LR schedule for THIS optimizer step (so the current update uses the intended LR)
                self.learning_rate_scheduler.step()
                self.optimizer.step()


            # Log to wandb (per step)
            if self.use_wandb:
                lrs = self.learning_rate_scheduler.get_last_lr()  # one per param group

                # Keep only the useful ones:
                # - main: usually the "other" weights group (last group)
                # - day: optional, if you care about the day-specific group (typically index 1)
                log_dict = {
                    "train/loss": float(loss.detach().item()),
                    "train/grad_norm": float(grad_norm),
                    "lr/main": float(lrs[-1]),
                }
                if len(lrs) > 1:
                    log_dict["lr/day"] = float(lrs[1])

                wandb.log(log_dict, step=i)

            
            # Save training metrics 
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Incrementally log training progress
            if i % self.args['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: ' +
                        f'loss: {(loss.detach().item()):.2f} ' +
                        f'grad norm: {grad_norm:.2f} '
                        f'time: {train_step_duration:.3f}')

            # Incrementally run a test step
            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")
                
                # Calculate metrics on val data
                start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - start_time


                # Log info 
                wer_tag = str(self.eval_cfg.get("wer_tag", "1gram"))
                wer_key = f"avg_WER_{wer_tag}"
                wer_val = val_metrics.get(wer_key, float("nan"))
                wer_n = int(val_metrics.get("wer_num_trials", 0))

                self.logger.info(
                    f'Val batch {i}: '
                    f'PER (avg): {val_metrics["avg_PER"]:.4f} '
                    f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} '
                    f'WER({wer_tag}): {wer_val:.2f}% (n={wer_n}) '
                    f'time: {val_step_duration:.3f}'
                )

                self.logger.info(
                    f'WER lens: avg_true_words={val_metrics.get("wer_avg_true_words", float("nan")):.2f} '
                    f'avg_pred_words={val_metrics.get("wer_avg_pred_words", float("nan")):.2f} '
                    f'max_pred_words={int(val_metrics.get("wer_max_pred_words", 0))}'
                )


                
                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_PERs'].keys():
                        self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}")

                # Save metrics 
                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                if self.use_wandb:
                    wer_tag = str(self.eval_cfg.get("wer_tag", "1gram"))
                    wer_key = f"avg_WER_{wer_tag}"

                    log_payload = {
                        "val/PER": float(val_metrics["avg_PER"]),
                        "val/loss": float(val_metrics["avg_loss"]),
                        # Single unified metric name across runs (1-gram or 5-gram)
                        "val/WER": float(val_metrics.get(wer_key, float("nan"))),
                    }

                    wandb.log(log_payload, step=i)



                # Determine if new best day. Based on if PER is lower, or in the case of a PER tie, if loss is lower
                # Prefer WER-based checkpointing when available; fallback to PER/loss otherwise
                wer_tag = str(self.eval_cfg.get("wer_tag", "1gram"))
                wer_key = f"avg_WER_{wer_tag}"
                cur_wer = float(val_metrics.get(wer_key, float("nan")))
                use_wer = self.compute_wer and np.isfinite(cur_wer)

                new_best = False
                if use_wer:
                    if cur_wer < self.best_val_WER:
                        self.logger.info(f"New best val WER({wer_tag}) {self.best_val_WER:.2f}% --> {cur_wer:.2f}%")
                        self.best_val_WER = cur_wer
                        # keep these for reference
                        self.best_val_PER = float(val_metrics["avg_PER"])
                        self.best_val_loss = float(val_metrics["avg_loss"])
                        new_best = True
                else:
                    # fallback: PER primary, loss as tie-break
                    if val_metrics['avg_PER'] < self.best_val_PER:
                        self.logger.info(f"New best val PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                        self.best_val_PER = float(val_metrics['avg_PER'])
                        self.best_val_loss = float(val_metrics['avg_loss'])
                        new_best = True
                    elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss):
                        self.logger.info(f"New best val loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                        self.best_val_loss = float(val_metrics['avg_loss'])
                        new_best = True


                if new_best:

                    # Checkpoint if metrics have improved 
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        best_wer_to_save = self.best_val_WER if np.isfinite(self.best_val_WER) else None
                        self.save_model_checkpoint(
                            f'{self.args["checkpoint_dir"]}/best_checkpoint',
                            self.best_val_PER,
                            self.best_val_loss,
                            best_wer_to_save,
                        )



                    # save validation metrics to pickle file
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f) 

                    val_steps_since_improvement = 0
                    
                else:
                    val_steps_since_improvement +=1

                cur_wer_to_save = float(val_metrics.get(wer_key, float("nan")))
                cur_wer_to_save = cur_wer_to_save if np.isfinite(cur_wer_to_save) else None

                self.save_model_checkpoint(
                    f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}',
                    val_metrics['avg_PER'],
                    val_metrics['avg_loss'],
                    cur_wer_to_save,
                )



                # Early stopping 
                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break
                
        # Log final training steps 
        training_duration = time.time() - train_start_time


        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model 
        if self.args['save_final_model']:
            self.save_model_checkpoint(
                f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}',
                val_PERs[-1],
                val_losses[-1],
            )


        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses 
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results

        if self.use_wandb:
            wandb.finish()



        return train_stats



    def validation(self, loader, return_logits = False, return_data = False):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()

        metrics = {}
        
        # Record metrics
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        total_edit_distance = 0.0
        total_seq_length = 0.0

        self._val_step_count += 1

        wer_every = int(self.eval_cfg.get("wer_every_val_steps", 1))
        do_wer_now = self.compute_wer and (self._lm is not None) and (wer_every > 0) and ((self._val_step_count % wer_every) == 0)
        wer_max_trials = int(self.eval_cfg.get("wer_max_trials", 64))
        wer_collected = []  # list of (logits_tc, true_sentence)

        # Calculate PER for each specific day
        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):        

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Determine if we should perform validation on this batch
            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    ps = int(self.args["model"]["patch_size"])
                    st = int(self.args["model"]["patch_stride"])

                    if ps > 0:
                        if st <= 0:
                            raise ValueError(f"Invalid patch_stride={st} with patch_size={ps}")
                        adjusted_lens = torch.div((n_time_steps - ps), st, rounding_mode="floor") + 1
                        adjusted_lens = adjusted_lens.to(torch.int32)

                    else:
                        adjusted_lens = n_time_steps.to(torch.int32)


                    logits = self.model(features, day_indicies)

                                        # Collect a subset for WER computation (expensive)
                    if do_wer_now and (len(wer_collected) < wer_max_trials):
                        for b in range(logits.shape[0]):
                            if len(wer_collected) >= wer_max_trials:
                                break

                            T = int(adjusted_lens[b].item())
                            logits_tc = logits[b, :T, :].detach().cpu().float().numpy()

                            true_sentence = _decode_transcription_to_str(batch["transcriptions"][b])
                            wer_collected.append((logits_tc, true_sentence))

                            if len(wer_collected) == 1:
                                self.logger.info(f"WER debug GT example: {true_sentence}")

    
                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)


                # Calculate PER per day and also avg over entire validation set
                batch_edit_distance = 0 
                decoded_seqs = []
                for iterIdx in range(logits.shape[0]):
                    decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(),dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0])

                    trueSeq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )
            
                    batch_edit_distance += editdistance.eval(decoded_seq.tolist(), trueSeq.tolist())

                    decoded_seqs.append(decoded_seq)

            day = batch['day_indicies'][0].item()
                
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()


            total_edit_distance += float(batch_edit_distance)
            total_seq_length += float(torch.sum(phone_seq_lens).item())


            # Record metrics
            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

            # total_seq_length puede ser tensor o float/int según cómo lo vayas acumulando
            if isinstance(total_seq_length, torch.Tensor):
                total_seq_length = total_seq_length.item()

        # --- finalize PER/loss ---
        metrics["day_PERs"] = day_per

        if total_seq_length == 0:
            metrics["avg_PER"] = float("nan")
        else:
            metrics["avg_PER"] = float(total_edit_distance / float(total_seq_length))

        metrics["avg_loss"] = float(np.mean(metrics["losses"])) if len(metrics["losses"]) > 0 else float("nan")

        # --- finalize WER (local LM) ---
        wer_tag = str(self.eval_cfg.get("wer_tag", "1gram"))
        primary_key = f"avg_WER_{wer_tag}"

        # Defaults
        metrics[primary_key] = float("nan")
        metrics["avg_WER_1gram"] = float("nan")
        metrics["avg_WER_5gram"] = float("nan")
        metrics["wer_num_trials"] = 0
        metrics["wer_avg_true_words"] = float("nan")
        metrics["wer_avg_pred_words"] = float("nan")
        metrics["wer_max_pred_words"] = 0

        if do_wer_now and len(wer_collected) > 0 and (self._lm is not None):
            # Decide which decoder is PRIMARY:
            # - if wer_tag==5gram and _lm_5gram exists => primary is 5gram decoder
            # - else => primary is _lm (whatever lm_dir points to)
            use_5_as_primary = (wer_tag == "5gram") and (self._lm_5gram is not None)
            primary_dec = self._lm_5gram if use_5_as_primary else self._lm
            secondary_dec = self._lm if use_5_as_primary else self._lm_5gram
            secondary_key = "avg_WER_1gram" if use_5_as_primary else "avg_WER_5gram"

            total_ed_primary = 0
            total_ed_secondary = 0
            total_words = 0

            true_lens = []
            pred_lens_primary = []

            debug_examples = int(self.eval_cfg.get("wer_debug_examples", 2))

            for logits_tc, true_sentence in wer_collected:
                lp = torch.from_numpy(logits_tc).float()
                log_probs_tc = torch.log_softmax(lp, dim=-1).cpu().numpy()

                pred_primary = primary_dec.decode_from_logits(log_probs_tc, input_is_log_probs=True)

                pred_secondary = None
                if secondary_dec is not None:
                    pred_secondary = secondary_dec.decode_from_logits(log_probs_tc, input_is_log_probs=True)

                true_clean = _normalize_for_wer(true_sentence)
                pred_clean_primary = _normalize_for_wer(pred_primary)
                pred_clean_secondary = _normalize_for_wer(pred_secondary) if pred_secondary is not None else ""

                true_words = true_clean.split()
                pred_words_primary = pred_clean_primary.split()
                pred_words_secondary = pred_clean_secondary.split() if pred_secondary is not None else None

                true_lens.append(len(true_words))
                pred_lens_primary.append(len(pred_words_primary))

                if debug_examples > 0:
                    if pred_secondary is not None:
                        self.logger.info(
                            f"WER debug example\n"
                            f"  GT : {true_clean}\n"
                            f"  PR : {pred_clean_primary}\n"
                            f"  SC : {pred_clean_secondary}"
                        )
                    else:
                        self.logger.info(
                            f"WER debug example\n"
                            f"  GT : {true_clean}\n"
                            f"  PR : {pred_clean_primary}"
                        )
                    debug_examples -= 1

                if len(true_words) == 0:
                    continue

                total_ed_primary += editdistance.eval(true_words, pred_words_primary)
                if pred_words_secondary is not None:
                    total_ed_secondary += editdistance.eval(true_words, pred_words_secondary)
                total_words += len(true_words)

            if total_words > 0:
                primary_wer = 100.0 * float(total_ed_primary) / float(total_words)
                metrics[primary_key] = primary_wer

                # Also fill convenience slots when they match reality
                if wer_tag == "1gram":
                    metrics["avg_WER_1gram"] = primary_wer
                if wer_tag == "5gram":
                    metrics["avg_WER_5gram"] = primary_wer

                if secondary_dec is not None:
                    metrics[secondary_key] = 100.0 * float(total_ed_secondary) / float(total_words)

            metrics["wer_num_trials"] = int(len(wer_collected))
            metrics["wer_avg_true_words"] = float(np.mean(true_lens)) if true_lens else float("nan")
            metrics["wer_avg_pred_words"] = float(np.mean(pred_lens_primary)) if pred_lens_primary else float("nan")
            metrics["wer_max_pred_words"] = int(np.max(pred_lens_primary)) if pred_lens_primary else 0



        return metrics