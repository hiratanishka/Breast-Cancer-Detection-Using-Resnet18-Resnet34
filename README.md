# Breast Cancer Detection Using ResNet18, ResNet34 and Hybrid ResNet Ensemble

This repository contains the implementation and experimental results of deep learning models for breast cancer detection using histopathological images.

The project evaluates three architectures:

- ResNet18
- ResNet34
- Hybrid ResNet18 + ResNet34 Ensemble

The models were trained using transfer learning on the Histopathologic Cancer Detection dataset.

---

## Publication

This work is associated with the published research paper:

### Breast Cancer Detection Using Resnet18 & Resnet34

**Authors:**  
Tanishka Hira, Anshita Jain, Versha Sharma, Shweta Jindal, Richa Yadav

Published in:

**Proceedings of International Conference on Artificial Intelligence and Networks (ICAIN 2025)**  
Lecture Notes in Networks and Systems, Springer.

### Paper Links

- Springer: https://link.springer.com/chapter/10.1007/978-3-032-24926-5_7
- ResearchGate: https://www.researchgate.net/publication/405257632_Breast_Cancer_Detection_Using_Resnet18_Resnet34

### DOI

https://doi.org/10.1007/978-3-032-24926-5_7

---

## Dataset

The experiments use the **Histopathologic Cancer Detection** dataset available on Kaggle.

The dataset contains histopathological image patches labelled for the presence or absence of metastatic cancer.

The training set contains approximately **220,000 TIFF images**.

### Classes

- `0` - No tumor tissue
- `1` - Tumor tissue present

The dataset itself is not included in this repository because of its size.

---

## Methodology

Transfer learning was used with ImageNet-pretrained convolutional neural networks.

The following layers were fine-tuned:

- `layer3`
- `layer4`
- final fully connected (`fc`) layer

Earlier layers were frozen to retain pretrained visual features.

Images were resized to:

`224 x 224`

and normalized using ImageNet normalization values.

### Dataset Split

The dataset was divided into:

- 70% Training
- 30% Validation

---

## Models

### 1. ResNet18

A pretrained ResNet18 model was adapted for binary classification.

The final fully connected layer was replaced with:

```python
nn.Linear(num_features, 2)
```

Fine-tuning was performed on the deeper layers of the network.

---

### 2. ResNet34

ResNet34 was trained using the same transfer-learning strategy as ResNet18.

Its greater network depth allows the model to learn more complex hierarchical image features.

---

### 3. Hybrid ResNet18 + ResNet34

The hybrid model combines predictions from ResNet18 and ResNet34.

Both networks independently process the same image.

Their output logits are averaged:

```python
output = (output_resnet18 + output_resnet34) / 2
```

The final prediction is obtained from the averaged output.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | SGD |
| Learning Rate | 0.0005 |
| Momentum | 0.9 |
| Weight Decay | 5e-4 |
| Batch Size | 64 |
| Epochs | 10 |
| Loss Function | Cross Entropy Loss |
| Scheduler | CyclicLR |
| Base LR | 1e-6 |
| Maximum LR | 1e-3 |
| Scheduler Mode | triangular2 |

---

## Results

| Model | Accuracy | AUC |
|---|---:|---:|
| ResNet18 | 90.24% | 0.96 |
| ResNet34 | 90.88% | 0.97 |
| **Hybrid ResNet18 + ResNet34** | **91.77%** | **0.97** |

The hybrid ensemble achieved the highest overall classification accuracy.

Additional reported metrics for the hybrid model include:

- Precision: 91.32%
- Recall: 88.15%
- F1 Score: 89.70%

---

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Area Under the ROC Curve (AUC)

Result plots are included inside their respective model directories.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── ResNet18/
│   ├── resnet18.ipynb
│   ├── ConfusionMatrix.png
│   ├── ROC.png
│   ├── T_V_accuracy.png
│   └── T_V_loss.png
│
├── ResNet34/
│   ├── resnet34.ipynb
│   ├── ConfusionMatrix.png
│   ├── ROC.png
│   ├── T_V_accuracy.png
│   └── T_V_loss.png
│
└── HybridModel/
    ├── hybridmodel.ipynb
    ├── ConfusionMatrix.png
    ├── ROC.png
    ├── T_V_accuracy.png
    └── T_V_loss.png
```

---

## Running the Notebooks

The notebooks can be executed using Google Colab.

### 1. Open a notebook in Google Colab

Upload the required `.ipynb` file to Colab.

### 2. Enable GPU acceleration

Go to:

`Runtime -> Change runtime type -> GPU`

### 3. Access the Kaggle dataset

The notebooks use KaggleHub to access the Histopathologic Cancer Detection competition dataset.

Kaggle authentication and competition access may be required.

### 4. Run the notebook cells

Run the notebook sequentially to:

1. Load the dataset
2. Create the training and validation sets
3. Initialize the model
4. Train the model
5. Evaluate performance
6. Generate the result plots

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Pillow
- Scikit-learn
- Matplotlib
- Seaborn
- KaggleHub
- Google Colab

---

## Citation

If you use or reference this work, please cite:

```text
Hira, T., Jain, A., Sharma, V., Jindal, S., Yadav, R. (2026).
Breast Cancer Detection Using Resnet18 & Resnet34.
Proceedings of International Conference on Artificial Intelligence
and Networks (ICAIN 2025).
Lecture Notes in Networks and Systems.
Springer.

DOI: 10.1007/978-3-032-24926-5_7
```

---

## Authors

- Tanishka Hira
- Anshita Jain
- Versha Sharma
- Shweta Jindal
- Richa Yadav
