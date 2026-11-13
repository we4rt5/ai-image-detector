# AI-Generated Image Detector (PyTorch)

A binary image classifier that tells **real photos** from **AI-generated images**, built from scratch in PyTorch. The project compares a small CNN trained from scratch with transfer learning on a pretrained ResNet18.

## Dataset

[CIFAKE](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset): real 32×32 images from CIFAR-10 and AI-generated counterparts made with Stable Diffusion.
A balanced subset was used: **20,000 training** and **4,000 test** images (≈50% REAL / 50% FAKE).

## Approach

| Part | Notebook | What was done |
|---|---|---|
| 1 | `01_tinyvgg_baseline.ipynb` | Data pipeline (`ImageFolder` → `DataLoader`), TinyVGG CNN from scratch, custom training loop (`train_step` / `test_step` / `train`), loss curves, saving the model |
| 2 | `02_evaluation.ipynb` | Loading saved weights, confusion matrix, precision / recall, inspecting misclassified images, experiment: 10 vs 32 filters, prediction on a custom image |
| 3 | `03_transfer_learning.ipynb` | Transfer learning with ResNet18 (ImageNet weights): frozen backbone vs full fine-tuning |

## Results

| Model | Trainable params | Test accuracy | Train time (Colab T4) |
|---|---|---|---|
| TinyVGG from scratch (5 epochs) | 4.3K | 87.5% | ~1.5 min |
| ResNet18, frozen backbone (3 epochs) | 1K | 87.0% | 3 min |
| ResNet18, full fine-tuning (3 + 2 epochs) | 11.2M | **95.3%** | 6 min |

## Key findings

- **A frozen ImageNet backbone is no better than a tiny CNN trained from scratch** (87.0% vs 87.5%). ImageNet features describe *what* is in the image (cats, cars, planes), while AI-generated images differ in *low-level pixel artifacts*. Upscaling 32×32 images to 224×224 for ResNet also blurs those artifacts.
- **Full fine-tuning gives +8 percentage points** (95.3%): letting the whole network adapt its features to the "real vs generated" task matters more than simply using a bigger pretrained model.
- **Trade-off:** fine-tuned ResNet18 is the most accurate but ~2,600× larger than TinyVGG and slower at inference; TinyVGG runs instantly on CPU.
- **Limitation:** both models are trained only on 32×32 CIFAKE images (mostly animals and vehicles), so predictions on large real-world photos or modern generators are not reliable.

## How to run

The notebooks are designed for Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU). Each notebook downloads the dataset subset automatically.

Local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tech stack

Python · PyTorch · torchvision · scikit-learn · pandas · matplotlib · Hugging Face `datasets`

## Next steps

- Web demo (Gradio): upload an image and get a prediction
- Refactor notebook code into Python modules (`data.py`, `model.py`, `train.py`)
- Test on higher-resolution images from modern generators
