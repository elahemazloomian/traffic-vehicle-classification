# Reliable traffic-vehicle classification (8 classes, PyTorch)

Project 2 of the Maktab Sharif AI bootcamp. A camera sees vehicles on a road; the model must name the vehicle type. This repository is a clean rewrite of an earlier notebook that leaked test images into training. Everything here is built around three questions: *is the evaluation honest*, *which choices actually help*, and *what happens when the model sees something it was never trained on*.

**Classes (8):** `ambulance`, `autobus`, `kamyun` (large truck), `kamyunet` (small city truck), `minibus`, `savari` (car), `taxi`, `vanet` (pickup).
**Unknown class:** `neysan` (Nissan pickup). The model never trains on it, in any experiment or in the final model. It is used only to study how the system reacts to an unseen vehicle.

## Headline results

| Model (last epoch, never the best epoch) | Mean macro-F1 on val, 3 seeds | Range across seeds |
|---|---|---|
| SimpleCNN, 5 conv blocks (982k parameters) | 0.876 | 2.3 pts |
| ResNet18, feature extraction (4,104 trainable parameters) | 0.872 | 1.3 pts |
| **ResNet18, fine-tune `layer4` + new head (8.4M trainable)** | **0.934** | **0.6 pts** |
| ResNet18, fine-tune `layer3` + `layer4` + head (10.5M trainable) | 0.942 | 0.3 pts |

Scores are the mean of the last 5 epochs. The chosen model is ResNet18 fine-tuning `layer4`. The `layer3` variant was tried last and was **not** adopted, see "Decisions".

**Test set, evaluated once** (ResNet18 fine-tune `layer4`, trained on the training split only):

| Test setting | Accuracy | Macro-F1 |
|---|---|---|
| 8 known classes only (361 images) | **0.942** | **0.938** |
| Known classes + 302 Nissan images, no "unknown" option | 0.513 | - |
| Known classes + 302 Nissan images, confidence threshold 0.99 | 0.753 | 0.797 (9 outcomes) |

Per-class test scores (8 classes only):

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| ambulance | 1.000 | 0.960 | 0.980 | 50 |
| autobus | 0.962 | 1.000 | 0.980 | 50 |
| kamyun | 0.947 | 0.750 | 0.837 | 48 |
| kamyunet | 0.803 | 0.942 | 0.867 | 52 |
| minibus | 1.000 | 0.980 | 0.990 | 50 |
| savari | 0.926 | 1.000 | 0.962 | 50 |
| taxi | 1.000 | 0.960 | 0.980 | 50 |
| vanet | 0.909 | 0.909 | 0.909 | 11 |

The test score agrees with validation (val accuracy about 0.937), so the choices made on validation did not overfit it. This is one model and one run, not a confidence interval.

## Data and leakage prevention

- Provided folders: `train`, `test`, `unclean`. `unclean` holds the same 8 classes plus the Nissan class. The data itself is **not** in Git (`data/` is ignored).
- All images are RGB. A green rectangle is burned into every image. About 24% have a bright bottom edge (taxis about 40%, buses about 15%); this is a weak class hint and was left as is.
- **Test is frozen.** Train and unclean images that are near-duplicates of a test image (perceptual-hash distance <= 4) are dropped. Identical images with different labels are dropped; identical images with the same label keep only one copy.
- **Grouped split.** Near-duplicate images (distance <= 4) form a group and a whole group goes to train or validation, never both. The split is stratified by class (about 20% validation).
- Manual cleaning (5 Oct): Nissan pickups hidden inside `vanet` were moved to the unknown pool, and a few clearly wrong truck/car labels were moved. Every move is logged in `reports/data_changes.csv`.

| Class | train | val | test |
|---|---|---|---|
| ambulance | 175 | 44 | 50 |
| autobus | 191 | 48 | 50 |
| kamyun | 177 | 44 | 48 |
| kamyunet | 214 | 53 | 52 |
| minibus | 180 | 45 | 50 |
| savari | 202 | 51 | 50 |
| taxi | 196 | 49 | 50 |
| vanet | 61 | 16 | 11 |
| **total** | 1396 | 350 | 361 |

The unknown (Nissan) pool has 604 images. It is split by near-duplicate groups into two halves of 302 images: one half chooses the confidence threshold, the other half is only used for reporting.

## Training recipe

AdamW (lr 1e-3, weight decay 1e-4), batch 32, ImageNet normalisation, mild augmentation (horizontal flip, rotation 10 degrees, 5% translation, scale 0.9 to 1.1, brightness/contrast; **no hue change** because colour is a class cue, for example yellow taxis). **Fixed recipe:** 40 epochs, cosine learning-rate decay, no early stopping, keep the last epoch. Nothing depends on the validation set while training, so scores are less noisy and the same recipe can be reused for the final model. An earlier recipe with early stopping gave a 3-seed mean of 0.808 with a 7.1-point range, mostly because early stopping triggered at random times.

The seed is fixed (42, plus 43 and 44 for repeats). Identical settings reproduce identical results on the same GPU.

## Why global average pooling

A `Flatten` layer after the last pooling step needs a huge dense layer: for example, a 64x56x56 feature map flattened into a 128-unit layer has about 25.7 million parameters, which overfits a dataset this small. Global average pooling reduces every channel to one number, so the classifier head of our CNN is a single small linear layer. Batch normalisation after every convolution stabilises and speeds up training.

## Environment

Python 3.11 in an isolated environment, PyTorch with CUDA 12.1, one NVIDIA GTX 1650 laptop GPU. One training run of 40 epochs takes about 8 to 12 minutes.

## Experiments and what they tell us

All numbers are the mean macro-F1 of the last 5 epochs. "Gap" is train accuracy minus validation accuracy.

**Three seeds per arm (mean of 3):**

| Experiment | Mean F1 | Gap | Reading |
|---|---|---|---|
| SimpleCNN base | 0.876 | 7.8 pts | reference |
| + dropout 0.3 | 0.876 | 5.6 pts | same F1, less overfitting, smaller seed spread |
| BCE loss instead of cross-entropy | 0.876 | 7.4 pts | no measurable difference; cross-entropy kept |
| Imbalanced (minibus and taxi cut to 20%), random batches | 0.829 | - | cutting data costs about 4.7 pts |
| Imbalanced, `BalancedBatchSampler` | 0.853 | - | about 2.4 pts better, see below |

**One seed each (noise is about +-1.2 pts, so only large gaps count):**

| Change vs base (0.867, seed 42) | F1 | Reading |
|---|---|---|
| no horizontal flip | 0.878 | within noise: flipping is not harmful |
| weight decay 1e-3 / 0 | 0.868 / 0.867 | within noise |
| no augmentation | 0.854 | F1 within noise, but the gap grows from 8.6 to 12.3: augmentation reduces overfitting |
| average pooling | 0.832 | max pooling is better |
| constant learning rate | 0.812 | cosine decay clearly matters |

**Balanced batches** (equal number of images per class in every batch, small classes repeated). On the imbalanced data, per-class recall averaged over 3 seeds, random vs balanced: `minibus` 0.64 to 0.78, `vanet` 0.56 to 0.71, `kamyun` 0.73 to 0.77, `taxi` 0.89 to 0.89, `kamyunet` 0.81 to 0.78, `savari` 0.99 to 0.97. Per seed the F1 difference was -0.4, +2.5 and +5.2 points, and the ranges overlap, so this is a positive sign, not proof. The sampler helps the small classes at a small cost to the large ones.

**Transfer learning.** Feature extraction (only the new head trains, frozen BatchNorm layers kept in eval mode) is no better than the small CNN, but overfits much less (gap 2.6 pts). Fine-tuning `layer4` is clearly better (+5.8 pts over the small CNN, with very small seed spread).

## Error analysis

- The dominant confusion is `kamyun` vs `kamyunet`. On test, 12 of 48 `kamyun` images were called something else (recall 0.75) and `kamyunet` precision is 0.80. On validation the same pair accounted for most mistakes in every run.
- `vanet` is small (61 train, 16 val, 11 test), so its scores swing a lot: one validation image moves recall by 6 points and one test image by 9. Most seed-to-seed variation came from `vanet` and `kamyun`.
- Some of the most confident mistakes look like **label noise** between `kamyun` and `kamyunet` (for example a large truck with a Scania badge labelled `kamyunet`). This is an impression from looking at the pictures, not proof. During a manual review, 6 training images in `kamyun` looked like `kamyunet` (file IDs 214831926, 215099946, 215635130, 216133020, 217219987, 216977256). They were **not** moved, because moving them would have forced re-running all experiments; they are listed here for transparency.
- Merging `kamyun` and `kamyunet` would remove the main confusion, but the task is 8-class, so the project stays 8-class. Merging is an analysis idea only.
- The row-normalised confusion matrices and the 12 most confident mistakes per run are saved by `evaluate.py` in `reports/figures/`.

## Unknown vehicles (Nissan)

The model has 8 outputs, so it must always name one of them. To allow "I am not sure", an image is marked `needs_review` when its highest softmax probability is below a threshold. The threshold (0.99) was chosen on validation only (known validation images plus the first Nissan half) by maximising macro-F1 over 9 outcomes (8 classes + unknown).

| | Nissan half used to choose | Nissan half held out |
|---|---|---|
| Correctly rejected as unknown | 64.9% | 63.9% |
| Still called `vanet` with confidence >= 0.99 | 35.1% | 35.8% |
| Called `vanet` with no threshold at all | 87.4% | 90.7% |

Price of the threshold: 14% of known validation images go to `needs_review`; the images that stay are right 97% of the time. A Nissan pickup looks like a `vanet`, so a confidence threshold is a weak detector of unknown vehicles: about a third of Nissans are still accepted as `vanet` with very high confidence.

## Final model and how to use it

`checkpoints/final.pth` is the same recipe (ResNet18, fine-tune `layer4`, 40 epochs) trained on **all** known-class images (train + validation + test, no Nissans). It has no honest score of its own on our side, since it has seen every labelled image; the hidden test is its real check. Checkpoints are not in Git. Each checkpoint stores its own configuration, class list, image size and normalisation, so `predict.py` rebuilds everything from the file.

```
python -m src.predict --checkpoint checkpoints\final.pth --images path\to\image_or_folder --threshold 0.99
```

It writes `reports/predictions.csv` with the predicted class, its confidence, the second choice and a decision (`accept` or `needs_review`).

Caveat: the 0.99 threshold was chosen for the train-only model. The final model, trained on more data, is more confident: on the 604 Nissan images it rejected 46% instead of 64%. Treat 0.99 as a starting point and adjust `--threshold` if a different trade-off is needed.

## Decisions

- **Why ResNet18 fine-tuning of `layer4`.** It was the best model by a clear margin and the most stable across seeds. The model and the recipe were frozen before the test set was used.
- **The deeper variant was not adopted.** Fine-tuning `layer3` as well gave 0.942 against 0.934 (+0.9 pts) with a slightly smaller gap. The acceptance rule written before seeing the result was "at least +1 point averaged over 3 seeds", which it narrowly missed. Using it would also have meant looking at the test set a second time. It is reported here as a possible improvement.
- **Cross-entropy, not BCE.** No difference, and softmax probabilities are needed for the threshold.
- **Dropout 0.3** and **augmentation** help against overfitting and are kept in the SimpleCNN experiments.

## Limitations

- Small validation set (350 images); `vanet` has 16 validation and 11 test images, so its scores are very noisy.
- 63 weaker look-alike pairs (distance 6 or 8) still cross the train/validation boundary on this fixed-camera data, so validation is slightly optimistic. Test was frozen first, so it is not affected this way.
- Possible label noise between `kamyun` and `kamyunet` (see above).
- Three seeds per experiment, one GPU. Ranges are shown instead of confidence intervals.
- The unknown-vehicle threshold was tuned on `r_ft`, not on `final.pth`; Nissans are 46% of the combined test-plus-Nissan set, so that accuracy depends on this ratio. The hidden test may mix them differently.
- Perceptual-hash distances of 2 or more are unreliable on this data (the same background and camera), so only exact duplicates are removed automatically.
- The white bottom edge on many images has an unknown cause and was not removed.

## Reproduce

```
pip install -r requirements.txt
```

Place the data under `data/` (`train`, `test`, `unclean`), then run the pipeline in this order:

```
python -m src.audit
python -m src.duplicates
python -m src.clean_manifest
python -m src.split
python -m src.check_data
```

Experiments (each group finishes already-done runs quickly):

```
python -m src.run_ablations --only f_base f_base_seed43 f_base_seed44
python -m src.run_ablations --only r_fe r_fe_seed43 r_fe_seed44 r_ft r_ft_seed43 r_ft_seed44
python -m src.summarize_runs
```

Evaluation: `python -m src.evaluate --run r_ft`, `python -m src.unknown_eval --checkpoint checkpoints\r_ft.pth`, and the one-time test evaluation `python -m src.final_eval --checkpoint checkpoints\r_ft.pth`.

This pins the CUDA 12.1 build of PyTorch. On a machine without an NVIDIA GPU, delete the first line and the `+cu121` suffixes; everything also runs on CPU, only more slowly. Notebooks need Jupyter (for example the VS Code Jupyter extension).

## Repository layout

| Path | What it holds |
|---|---|
| `src/` | data loading, models (SimpleCNN, ResNet18), training, evaluation, experiment runner, prediction |
| `configs/config.yaml` | all defaults (override with `--set key=value`) |
| `reports/` | comparison table (`runs_table.md`), per-run histories, figures, data change log, thresholds |
| `notebooks/` | data audit, duplicates, split |
| `legacy/` | the old notebook, kept for reference only |
