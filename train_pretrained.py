
import os
from mmm.interactive import tasks, training, data, pipes
import torchvision.transforms.functional as F
import torch
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from torchvision import datasets
#from hparams import build_hparams
from pathlib import Path
import wandb


wandb.init()


# 1. Load pretrained encoder+squeezer
# --------------------------------------
device = torch.device("cpu")  # force CPU
model = NativeBlocks(MMM_MODELS[DEFAULT_MODEL], device_identifier=str(device))
encoder, squeezer = model["encoder"], model["squeezer"]
encoder.freeze_all_parameters()  # freeze encoder, only train squeezer+task



# 0. Weights and biases and cpu core initialization
# ------------------------------------------------------------------------------

if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: range(os.cpu_count() or 1)


def to_mmm(case):
    """Convert (PIL, label) -> dict with image+class"""
    pil_img, label = case
    return {"image": F.to_tensor(pil_img.convert("RGB")), "class": label}

def make_cohort(train_ds, val_ds):
    return data.TrainValCohort(
        data.TrainValCohort.Config(batch_size=(8,8), num_workers=2),
        train_ds=data.ClassificationDataset(
            train_ds,
            src_transform=to_mmm,
            batch_transform= pipes.Alb(pipes.get_histo_augs()),
        
            class_names=train_ds.classes
        ),
        val_ds=data.ClassificationDataset(
            val_ds,
            src_transform=to_mmm,
            class_names=val_ds.classes
        )
    )



def train():

 


 # 2. Prepare dataset
 # -----------------------------

 train_ds = datasets.ImageFolder("C:/Users/nroy/Pictures/Dataset_Split/train")
 val_ds   = datasets.ImageFolder("C:/Users/nroy/Pictures/Dataset_Split/val")

 
 cohort = make_cohort(train_ds, val_ds)

 # -----------------------------
 # 3. Define classification task
 # -----------------------------
 task = tasks.ClassificationTask(
    hidden_dim=squeezer.get_hidden_dim(),
    args=tasks.ClassificationTask.Config(module_name="my_pretrained_cls"),
    cohort=cohort,
 )

 # -----------------------------
 # 4. Build trainer
 # -----------------------------


 trainer = training.MTLTrainer(
    training.MTLTrainer.Config(
        checkpoint_cache_folder="trainer_checkpoints",
        max_epochs= 1,
        train_device=str(device),
    ),
    experiment_name="pretrained_cpu_exp_2",
    clear_checkpoints=True,
    ).add_shared_blocks([encoder, squeezer])


 trainer.add_mtl_task(task)
 

 # -----------------------------
 # 5. Train
 # -----------------------------
 trainer.fit()



if __name__ == "__main__":
    train()






 
 