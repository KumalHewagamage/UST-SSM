# UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling 🚀
This repository contains the official PyTorch implementation for the paper: **"UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling"**, accepted at ICCV 2025.

![pipeline](assets/pipeline.png)

-----

## 📜 Introduction

Point cloud videos capture dynamic 3D motion while reducing the effects of lighting and viewpoint variations, making them highly effective for recognizing subtle and continuous human actions. Although Selective State Space Models (SSMs) have shown good performance in sequence modeling with linear complexity, the spatio-temporal disorder of point cloud videos hinders their unidirectional modeling when directly unfolding the point cloud video into a 1D sequence through temporally sequential scanning. To address this challenge, we propose the Unified Spatio-Temporal State Space Model (UST-SSM), which extends the latest advancements in SSMs to point cloud videos. Specifically, we introduce Spatial-Temporal Selection Scanning (STSS), which reorganizes unordered points into semantic-aware sequences through prompt-guided clustering, thereby enabling the effective utilization of points that are spatially and temporally distant yet similar within the sequence. For missing 4D geometric and motion details, Spatio-Temporal Structure Aggregation (STSA) aggregates spatio-temporal features and compensates. To improve temporal interaction within the sampled sequence, Temporal Interaction Sampling (TIS) enhances fine-grained temporal dependencies through non-anchor frame utilization and expanded receptive fields. Experimental results on the MSR-Action3D, NTU RGB+D, and Synthia 4D datasets validate the effectiveness of our method.

-----

## 🛠️ Getting Started

### Prerequisites

  * Python 3.8+
  * PyTorch 1.12.0+
  * CUDA 11.3+

### Installation

1.  **Clone the repository:**

2.  **Install Python dependencies:**
    We recommend using a virtual environment (e.g., conda or venv).

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Mamba and Causal Conv1d:**

    ```bash
    pip install causal-conv1d mamba-ssm
    ```

4.  **Compile custom CUDA layers:**
    Our model relies on custom CUDA operators for PointNet++ and k-Nearest Neighbors (kNN).

      * **PointNet++ Layers:**
        ```bash
        cd modules/
        python setup.py install
        cd ..
        ```
      * **kNN for PyTorch:**
        ```bash
        pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
        ```

    Build notes & troubleshooting
    -----------------------------

    If you run into build or import errors when compiling the custom CUDA layers, here are the exact checks and commands that worked in this repository's setup and common fixes:

    - Activate the environment and check `nvcc` and PyTorch CUDA:

    ```bash
    conda activate ssm
    which nvcc || echo "nvcc not found in PATH"
    nvcc --version || true
    python -c "import torch; print('torch.version.cuda =', torch.version.cuda, 'torch.cuda.is_available() =', torch.cuda.is_available())"
    ```

    - If `nvcc` is not found in the conda env but is available on the system (for example `/usr/local/cuda/bin/nvcc` or `/usr/bin/nvcc`), point the build to the system CUDA (no system CUDA install/upgrade required):

    ```bash
    export PATH=/usr/local/cuda/bin:$PATH    # or /usr/bin if nvcc is there
    export CUDA_HOME=/usr/local/cuda         # path to your system CUDA
    ```

    - Install the `KNN_CUDA` wheel (used by this repo):

    ```bash
    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
    ```

    - Build the PointNet++ CUDA extension (from the repo):

    ```bash
    cd modules
    python setup.py install
    cd ..
    ```

    - Common import error: `ImportError: libc10.so: cannot open shared object file`

    If you see this when trying `import pointnet2._ext`, set `LD_LIBRARY_PATH` so the loader can find the conda env libraries and PyTorch libs:

    ```bash
    export CONDA_PREFIX=$(conda info --base)/envs/ssm   # optional helper; replace with your env path if needed
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
    python -c "import pointnet2._ext as _ext; print('loaded', _ext)"
    ```

    Note: on this machine we used `/home/avishka/anaconda3/envs/ssm` for `CONDA_PREFIX` in the commands above.

    - Minor CUDA-version mismatch warning

    You may see a non-fatal warning like "detected CUDA version (12.0) mismatches the version that was used to compile PyTorch (12.1)". In many cases that is harmless and the extension will still compile successfully; if you see hard failures then either install a PyTorch wheel that matches your system CUDA or point the environment to a matching `nvcc`.

    If you prefer not to compile CUDA ops, you can try installing a prebuilt `pointnet2` package (may not match exactly):

    ```bash
    pip install git+https://github.com/erikwijmans/Pointnet2_PyTorch.git
    ```

    If that succeeds, you can skip `python setup.py install`. Otherwise follow the build steps above.


For a detailed environment setup, please refer to the `requirements.yml` file.

-----

## 📊 Datasets

You will need to download and preprocess the datasets before training and evaluation.

### MSR-Action3D

1.  **Download** the dataset from [Google Drive](https://drive.google.com/file/d/1djwAK3oZTAIFbCz531eClxINmsZgGO_H/view?usp=sharing).
2.  Extract the `.zip` file to get `Depth.rar`, and then extract the depth maps.
3.  **Preprocess** the depth maps into point clouds by running the script:
    ```bash
    python scripts/preprocess_file.py --input_dir /path/to/your/Depth --output_dir /path/to/processed_data --num_cpu 11
    ```

### NTU RGB+D

1.  **Download** the dataset from the [official website](https://rose1.ntu.edu.sg/dataset/actionRecognition/). You will need to request access.
2.  After downloading, **convert** the depth maps to point cloud data using our script:
    ```bash
    python scripts/depth2point4ntu120.py --data_path /path/to/your/ntu_dataset
    ```

### Synthia 4D

1.  **Download** the dataset from the [official project page](http://cvgl.stanford.edu/data2/Synthia4D.tar).
2.  Extract the `.tar` file. The data should be ready for use without further preprocessing.

-----

## 🚀 Usage

### Training

To train the UST-SSM model on a dataset, use the following command structure. Make sure to specify the dataset path and the configuration file.

```bash
python train.py --config cfgs/msr-action3d_config.yaml --data_path /path/to/processed_data
```

### Evaluation

To evaluate a trained model, provide the path to your model checkpoint (`.pth` file).

```bash
python test.py --config cfgs/msr-action3d_config.yaml --data_path /path/to/processed_data --checkpoint /path/to/your/model.pth
```

-----

## 🙏 Acknowledgement

This work builds upon the excellent codebase of [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution). We thank the authors for making their code publicly available. We are also grateful for the advancements in State Space Models, particularly [Mamba](https://github.com/state-spaces/mamba).

-----

## ✍️ Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{li2025ust,
  title={UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling},
  author={Li, Peiming and Wang, Ziyi and Yuan, Yulin and Liu, Hong and Meng, Xiangming and Yuan, Junsong and Liu, Mengyuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6738--6747},
  year={2025}
}
```
