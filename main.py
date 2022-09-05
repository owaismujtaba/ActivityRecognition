from  utils import create_dataset, load_processed_data
from train import trainer

print("1: Create Dataset 2: Train Model")
choice = 2

if choice ==1:
    create_dataset()
if choice == 2:
    trainer()
