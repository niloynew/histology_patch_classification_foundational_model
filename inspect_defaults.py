from mmm.interactive import training, tasks
from mmm.labelstudio_ext.NativeBlocks import MMM_MODELS, DEFAULT_MODEL
import pprint
import torch

def main():
    # ------------------------------
    # 1. Trainer defaults
    # ------------------------------
    cfg = training.MTLTrainer.Config()

    print("\n=== FULL DEFAULT TRAINER CONFIG ===")
    pprint.pprint(cfg.__dict__)

    print("\n=== Optimizer defaults ===")
    print(cfg.optim)

    print("\n=== Train loop defaults ===")
    print(cfg.mtl_train_loop)

    print("\n=== Validation loop defaults ===")
    print(cfg.mtl_val_loop)


    # ------------------------------
    # 2. Model defaults
    # ------------------------------
    print("Default model:", DEFAULT_MODEL)
    print("\nAll available models:\n")
    print(MMM_MODELS[DEFAULT_MODEL])

    for key in MMM_MODELS.keys():
     print("-", key)

    # ------------------------------
    # 3. ClassificationTask defaults
    # ------------------------------
    print("\n=== ClassificationTask defaults ===")
    cfg = tasks.ClassificationTask.Config(module_name="dummy_cls")
    print(cfg)

        

if __name__ == "__main__":
    main()