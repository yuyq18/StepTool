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

