"""
q1/config.py  —  Central configuration for all Q1 experiments.
All other modules import from here; change values here only.
"""

import os
import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data ──────────────────────────────────────────────────────────────────────
NUM_CLASSES = 100
DATA_DIR    = "./data"

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle',
    'bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle',
    'caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch',
    'crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest',
    'fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower',
    'leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear',
    'pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum',
    'rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew',
    'skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower',
    'sweet_pepper','table','tank','telephone','television','tiger','tractor','train',
    'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS       = 10
LR           = 3e-4
WEIGHT_DECAY = 1e-4

# ── LoRA grid ─────────────────────────────────────────────────────────────────
LORA_DROPOUT = 0.1
RANKS        = [2, 4, 8]
ALPHAS       = [2, 4, 8]

# ── WandB ─────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "DLOps-A5-Q1"

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR  = "weights"
PLOTS_DIR = "plots"
LOGS_DIR  = "logs"
BEST_DIR  = os.path.join(SAVE_DIR, "BEST_MODEL")

# ── Workers ───────────────────────────────────────────────────────────────────
NUM_WORKERS = min(os.cpu_count() or 4, 16)
