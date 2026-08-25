"""The minimal CLIP wrapper used by the gradient-sharing experiment."""

import torch
import torch.nn.functional as F
from torch import nn


CLIP_MODEL = "openai/clip-vit-base-patch16"


def load_clip(model_name: str = CLIP_MODEL):
    from transformers import AutoTokenizer, CLIPModel

    model = CLIPModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


class TwoStageCLIP(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames, template):
        super().__init__()
        names = [self._classname_text(name) for name in classnames]
        tokens = tokenizer(
            [template.format(name) for name in names],
            padding=True,
            return_tensors="pt",
        )
        self.model = clip_model
        self.tokenizer = tokenizer
        self.template = template
        self.register_buffer("input_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tokens["attention_mask"], persistent=False)

    @staticmethod
    def _classname_text(classname):
        name = classname if isinstance(classname, str) else classname[0]
        return name.replace("_", " ")

    def encode_image(self, images):
        output = self.model.vision_model(pixel_values=images)
        return F.normalize(self.model.visual_projection(output.pooler_output), dim=-1)

    def encode_text(self):
        output = self.model.text_model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )
        return F.normalize(self.model.text_projection(output.pooler_output), dim=-1)

    def encode_classnames(self, classnames):
        names = [self._classname_text(name) for name in classnames]
        tokens = self.tokenizer(
            [self.template.format(name) for name in names],
            padding=True,
            return_tensors="pt",
        ).to(self.input_ids.device)
        output = self.model.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        return F.normalize(self.model.text_projection(output.pooler_output), dim=-1)

    def stage_one_logits(self, images):
        return self.model.logit_scale.exp() * self.encode_image(images) @ self.encode_text().t()

    def classifier_logits(self, images, classifier):
        classifier = F.normalize(classifier, dim=-1)
        return self.model.logit_scale.exp() * self.encode_image(images) @ classifier.t()
