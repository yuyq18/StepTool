# StepTool

## Environment Installation

1. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

    ```bash
    conda create -n steptool python=3.10
    ```

2. Activate the newly created environment.

    ```bash
    conda activate steptool
    ```

3. Install [pytorch](https://pytorch.org/get-started/locally/) according to your CUDA version and other required modules from pip.

```bash
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
```

Please make sure the version of GCC/G++ >= 9.0.0.

## Download the data

1. Download the compressed dataset from 

2. Uncompress the downloaded data_train.zip. The following command will directly extract data_train.zip into the .data_train/ directory

```bash
unzip data_train.zip
```


## SFT Training for base models

1. key parameters

`TRAIN_SET` and `MODEL_PATH`

2. run scripts

```bash
bash scripts/sft/train_qwen2.sh
bash scripts/sft/train_llama3-1.sh
```