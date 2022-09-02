A model to do activity estimation

2nd edition:

Update:
1. Add `wandb` to monitor training & testing
2. Add `cnt_loss`
3. Improve the speed of validation

Usage:

1. Use SlowFast repository to extract features for all the videos of Charades dataset 

2. Add current path to `PYTHONPATH`

    `export PYTHONPATH=$PYTHONPATH:$PWD`

3. Install requirements

    `pip install -r requirements`

4. Download pretrain model

    `bash ./models/download.sh`

5. Run the model

    `bash ./scripts/run.sh [PATH_TO_CONFIG] [INDEX_OF_GPU]`

    If you want to use `wandb` to monitor the training, set up `wandb` by 

    `wandb login`

    You can find more about `wandb` here: https://docs.wandb.ai/quickstart#1.-set-up-wandb
