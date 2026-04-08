"""
q1/model.py  —  ViT-S model builders: head-only baseline and LoRA variants.

Base checkpoint : WinKawaks/vit-small-patch16-224
LoRA targets    : query, key, value (attention projections)
"""

from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

from q1.config import CIFAR100_CLASSES, LORA_DROPOUT, NUM_CLASSES

_BASE_CKPT = "WinKawaks/vit-small-patch16-224"


def _load_vit(num_classes: int = NUM_CLASSES) -> ViTForImageClassification:
    """Load ViT-S with a fresh classification head for CIFAR-100."""
    return ViTForImageClassification.from_pretrained(
        _BASE_CKPT,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(CIFAR100_CLASSES)},
        label2id={c: i for i, c in enumerate(CIFAR100_CLASSES)},
    )


def build_baseline(num_classes: int = NUM_CLASSES):
    """
    Head-only baseline: freeze everything, unfreeze only the classifier head.
    Trainable params ≈ 38 500 (head weights + bias).
    """
    model = _load_vit(num_classes)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def build_lora(
    rank:    int,
    alpha:   int,
    dropout: float = LORA_DROPOUT,
    num_classes: int = NUM_CLASSES,
):
    """
    LoRA variant: low-rank updates on Q, K, V projections + trainable head.

    Args:
        rank:    LoRA rank r  (controls expressiveness vs parameter count)
        alpha:   LoRA scaling factor α
        dropout: dropout applied inside LoRA layers
    """
    base = _load_vit(num_classes)
    cfg  = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(base, cfg)
    # also unfreeze the classification head
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True
    return model
