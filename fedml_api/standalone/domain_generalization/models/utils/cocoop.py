# utils/cocoop.py

import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn

# Assuming clip and dassl are in your environment
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.optim import build_lr_scheduler, build_optimizer
from easydict import EasyDict as edict
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, "clips")
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Freeze encoders
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.training and label is not None:
            return F.cross_entropy(logits, label)

        return logits


def create_cocoop_cfg(args):
    cfg = edict()

    # MODEL
    cfg.MODEL = edict()
    cfg.MODEL.BACKBONE = edict()
    cfg.MODEL.BACKBONE.NAME = args.clip_backbone

    # TRAINER
    cfg.TRAINER = edict()
    cfg.TRAINER.COCOOP = edict()
    cfg.TRAINER.COCOOP.N_CTX = args.n_ctx
    cfg.TRAINER.COCOOP.CTX_INIT = args.ctx_init
    cfg.TRAINER.COCOOP.PREC = args.precision

    cfg.OPTIM = edict()
    cfg.OPTIM.NAME = "adam"
    cfg.OPTIM.LR = 0.0003
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9
    cfg.OPTIM.SGD_DAMPNING = 0
    cfg.OPTIM.SGD_NESTEROV = False
    cfg.OPTIM.RMSPROP_ALPHA = 0.99
    cfg.OPTIM.ADAM_BETA1 = 0.9
    cfg.OPTIM.ADAM_BETA2 = 0.999
    cfg.OPTIM.STAGED_LR = False
    cfg.OPTIM.NEW_LAYERS = ()
    cfg.OPTIM.BASE_LR_MULT = 0.1
    cfg.OPTIM.LR_SCHEDULER = "single_step"
    cfg.OPTIM.STEPSIZE = (-1,)
    cfg.OPTIM.GAMMA = 0.1
    cfg.OPTIM.MAX_EPOCH = 10
    cfg.OPTIM.WARMUP_EPOCH = -1
    cfg.OPTIM.WARMUP_TYPE = "linear"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5
    cfg.OPTIM.WARMUP_MIN_LR = 1e-5
    cfg.OPTIM.WARMUP_RECOUNT = True

    # cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = args.lr_clip
    cfg.OPTIM.MAX_EPOCH = args.epochs_clip
    # cfg.OPTIM.LR_SCHEDULER = "cosine"
    # cfg.OPTIM.WARMUP_EPOCH = 1

    return cfg


class CoCoOpTuner:
    """Encapsulates the logic for tuning a CoCoOp prompt learner."""

    def __init__(self, args, device):
        self.cfg = create_cocoop_cfg(args)
        self.device = device

    def _build_model(self, classnames):
        print("Building CustomCLIP model for tuning...")
        clip_model = load_clip_to_cpu(self.cfg)

        if self.cfg.TRAINER.COCOOP.PREC == "fp32":
            clip_model.float()

        model = CustomCLIP(self.cfg, classnames, clip_model)
        model.to(self.device)
        return model

    def tune(self, train_loader, classnames):
        model = self._build_model(classnames)

        # Only optimize the prompt_learner parameters
        optimizer = build_optimizer(model.prompt_learner, self.cfg.OPTIM)
        scheduler = build_lr_scheduler(optimizer, self.cfg.OPTIM)

        for epoch in range(self.cfg.OPTIM.MAX_EPOCH):
            model.train()
            loop = tqdm(
                train_loader,
                desc=f"CLIP Tuning Epoch {epoch+1}/{self.cfg.OPTIM.MAX_EPOCH}",
            )
            for batch in loop:
                # Assuming batch is a dict with 'img' and 'label'
                image, label = batch[0].to(self.device), batch[1].to(self.device)

                loss = model(image, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_postfix(loss=loss.item())

            scheduler.step()

        return model.prompt_learner.cpu().state_dict()
