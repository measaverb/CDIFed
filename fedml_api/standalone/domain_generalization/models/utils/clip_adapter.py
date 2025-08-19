# fedml_api/standalone/domain_generalization/models/utils/clip_adapter.py

import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from easydict import EasyDict as edict
from torch.nn import functional as F

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {"default": "a photo of a {}."}


def create_clip_cfg(args):
    cfg = edict()

    # MODEL
    cfg.MODEL = edict()
    cfg.MODEL.BACKBONE = edict()
    cfg.MODEL.BACKBONE.NAME = args.clip_backbone

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

    cfg.OPTIM.LR = args.lr_clip
    cfg.OPTIM.MAX_EPOCH = args.adapter_epochs

    return cfg


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="clips")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.2):
        super(Adapter, self).__init__()
        self.ratio = ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        adapter_features = self.fc(x)
        # Residual connection
        x = self.ratio * adapter_features + (1 - self.ratio) * x
        return x


class TextEncoder(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        temp = CUSTOM_TEMPLATES["default"]
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            self.clip_model.positional_embedding.device
        )
        text_features = self.clip_model.encode_text(prompts)
        return text_features


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder_base = clip_model.visual
        self.text_encoder = TextEncoder(classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        feature_dim = self.image_encoder_base.output_dim
        self.adapter = Adapter(feature_dim, 4).to(self.dtype)

    def encode_image(self, image):
        base_features = self.image_encoder_base(image.type(self.dtype))
        adapted_features = self.adapter(base_features)
        return adapted_features

    def forward(self, image):
        image_features = self.encode_image(image)
        text_features = self.text_encoder()

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits
