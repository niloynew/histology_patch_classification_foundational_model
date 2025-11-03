
import os
from mmm.interactive import tasks, training, data, pipes
import torchvision.transforms.functional as F
import torch
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from torchvision import datasets
from pathlib import Path
import wandb


if not os.path.exists("/deep_learning/output/niloy/fm_classification/wandb"):
    os.mkdir("/deep_learning/output/niloy/fm_classification/wandb")

os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/niloy/fm_classification/wandb")

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="fm_classification",
    dir="/deep_learning/output/niloy/fm_classification/wandb",
    # track hyperparameters and run metadata
    config={
    
    "architecture": "MMM",
    "dataset": "NCT-CRC-HE-100K",
    
    }
)


# 1. Load pretrained encoder+squeezer
# --------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # force CPU
model = NativeBlocks(MMM_MODELS[DEFAULT_MODEL], device_identifier=str(device))
encoder, squeezer = model["encoder"], model["squeezer"]
encoder.freeze_all_parameters()  # freeze encoder, only train squeezer+task



# 0. Weights and biases and cpu core initialization
# -----------------------------------------------------------------------------

def to_mmm(case):
    """Convert (PIL, label) -> dict with image+class"""
    pil_img, label = case
    return {"image": F.to_tensor(pil_img.convert("RGB")), "class": label}

def make_cohort(train_ds, val_ds):
    return data.TrainValCohort(
        data.TrainValCohort.Config(batch_size=(16,16), num_workers=4),
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
 train_ds = datasets.ImageFolder("/deep_learning/input/data/histology100K/Dataset_Split/train")
 val_ds   = datasets.ImageFolder("/deep_learning/input/data/histology100K/Dataset_Split/val")

 
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
        max_epochs= 50,
        train_device=str(device),
    ),
    experiment_name="pretrained_exp",
    clear_checkpoints=True,
    ).add_shared_blocks([encoder, squeezer])


 trainer.add_mtl_task(task)
 

 # -----------------------------
 # 5. Train
 # -----------------------------
 trainer.fit()



if __name__ == "__main__":
    train()   