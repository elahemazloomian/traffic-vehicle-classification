# Traffic Vehicle Classification with CNNs and PyTorch

This project focuses on building a robust, reproducible, and transparent image classification system for cropped traffic-camera vehicles. Beyond achieving high accuracy, the core objective is to audit data quality, handle label noise and out-of-distribution samples, prevent data leakage, and implement uncertainty-aware predictions.

---

## Project Structure

```text
traffic-vehicle-classification/
├── README.md                 # Project documentation and workflow rationale
├── requirements.txt          # Python dependencies (PyTorch, torchvision, imagehash, scikit-learn, etc.)
├── .gitignore                # Excludes raw datasets and model checkpoints from version control
├── dataset_cleaned/          # Cleaned and stratified training and validation splits
│   ├── train/                # 80% stratified training data
│   └── val/                  # 20% stratified validation data
├── notebooks/                # Exploratory data analysis and notebooks
├── src/                      # Core modular source code
│   ├── __init__.py
│   ├── audit.py              # Dataset audit script (file integrity, small image checks, pHash duplication)
│   ├── clean_and_split.py    # Automated data cleaning, duplicate removal, and stratified splitting
│   ├── dataset.py            # Dataset loaders, transforms, and custom samplers (planned)
│   ├── models.py             # CNN baseline and ResNet architectures (planned)
│   ├── train.py              # Training loops, loss functions, and logging (planned)
│   └── evaluate.py           # Evaluation metrics, confusion matrices, and error analysis (planned)
└── checkpoints/              # Saved model weights and training artifacts (ignored by git)




Phase 1: Data Audit and Leakage Prevention

Why Audit the Dataset?

    Before training any model, auditing the dataset is critical to ensure data integrity. Real-world traffic images often contain corrupted files, extreme aspect ratios, or low-resolution samples that can destabilize training.



Why Perceptual Hashing (pHash) and Hamming Distance?

    The Flaw of Exact Matching: Cryptographic hashes (like MD5) or exact filename matches fail when images undergo minor compression, resizing, or format shifts.

Perceptual Hashing (pHash):

    Generates a visual fingerprint based on image content, allowing us to capture semantic similarity.

Hamming Distance (Threshold <= 5): 

    Instead of requiring a 100% identical string match, we use Hamming distance to count differing bits between hashes. Setting a threshold (<= 5) successfully detects real duplicates and modified variants (due to compression or resizing) across train, test, and unclean splits.



Preventing Data Leakage

    Protecting the Test Set: Any overlapping or similar images between the training data and the frozen test set were identified and purged from the training pipeline. This guarantees that our final evaluation on the test set remains unbiased and truly unseen.



Phase 2: Handling Out-of-Distribution Data (neysan Class)


The Role of neysan: 
    The unclean split contains an extra class (neysan) alongside the 8 standard vehicle classes (ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet).

Why Exclude it from Training?
    Since our core classifier is trained on 8 specific classes, neysan acts as an out-of-distribution (OOD) / unseen class. Rather than forcing it into the training loop, it is intentionally reserved to test the model's uncertainty estimation and power a human-review mechanism (needs_review=True) for low-confidence predictions.


Phase 3: Data Cleaning and Stratified Splitting


Duplicate Removal: 
    Using the pHash audit with Hamming distance, redundant duplicates and cross-split leakages were safely filtered out.

Merging Clean Sources:
    Valid images from authorized splits (excluding neysan from training) were combined into a unified pool of 1,166 clean samples.

Stratified 80/20 Split:

    To maintain balanced class distributions across subsets, a stratified split with a fixed random seed (random_state=42) was applied.

    Training Set: 932 samples (used for model optimization and learning).

    Validation Set: 234 samples (used exclusively for hyperparameter tuning, scheduler decisions, and threshold selection, keeping the test set strictly frozen).




Today's Progress & Milestones (September 20, 2026)


Environment Configuration:
     Created an isolated Python 3.11 environment (`vehicle_env`) to resolve compatibility and dependency conflicts.

Hardware Acceleration (CUDA):
     Successfully configured and verified PyTorch with CUDA 12.1 support to run the deep learning pipeline on the NVIDIA GTX 1650 GPU.

Training Pipeline (`train.py`):
    Implemented a robust 5-epoch training loop using `CrossEntropyLoss` and the `AdamW` optimizer, targeting the fine-tuned ResNet-18 model's classification head (`fc`).

Added real-time tracking and printing of both Loss and Accuracy metrics for both Training and Validation phases.
Implemented smart model checkpointing to automatically save the best-performing weights (`best_model.pth`) based on validation accuracy (achieving over 80% validation accuracy).



Today's Progress & Milestones (September 26, 2026)

Fine-Tuning ResNet-18 (`train_finetune.py`)
    Unfreezing `layer4`: 
        In the initial feature extraction phase, the entire ResNet-18 backbone was frozen. To  adapt high-level features specifically to traffic vehicle characteristics (such as body lines, headlights, and windshields), we unfroze the deepest convolutional block (`layer4`) alongside the classification head.
    Differential Learning Rates:
         Applied a very small learning rate (`1e-5`) to the `layer4` backbone to preserve pretrained weights, and a larger learning rate (`1e-3`) to the classification head (`fc`) using the `AdamW` optimizer with weight decay (`1e-4`).

Building the Project-Required Simple CNN
    Requirement:
        Implemented a basic custom CNN featuring the required two convolution + activation + pooling blocks to serve as a foundational baseline from scratch.

Challenges of the Simple CNN & Designing `TrafficNetProficient`

    The Parameter Explosion & Overfitting Problem in Simple CNNs:

        In naive architectures using a standard `Flatten` layer after pooling (e.g., $64 \times 56 \times 56$), the spatial features are flattened into over 25 million parameters for a single dense layer. 
        On a limited vehicle dataset, this massive parameter count causes severe overfitting (near 100% training accuracy with poor validation generalization).

Solutions & Design of `TrafficNetProficient`:

    Global Average Pooling (`AdaptiveAvgPool2d`):

        Replaced heavy flattening with global average pooling to collapse spatial dimensions down to $1 \times 1$, drastically reducing parameter size and curbing overfitting.

    Batch Normalization (`BatchNorm2d`):
    
         Added after convolutions to stabilize internal covariate shifts, smooth the loss landscape, and accelerate convergence.

    Dropout ($p=0.4$):
  
        Integrated into the classifier head to randomly drop connections during training and enforce robust feature learning.

