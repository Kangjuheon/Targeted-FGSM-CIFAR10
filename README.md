## Targeted FGSM Attack on CIFAR-10 (ConvNeXt)

This project applies a **Targeted Fast Gradient Sign Method (FGSM)** adversarial attack to a **ConvNeXt-Tiny** model on the CIFAR-10 dataset.

## Description

Unlike untargeted attacks, **targeted FGSM** tries to **fool the model into predicting a specific target class**, not just a wrong one.

## Key Features

- Model: ConvNeXt-Tiny (pretrained on ImageNet)
- Attack: Targeted FGSM (`-eps * sign(gradient)`)
- Dataset: CIFAR-10 resized to 224×224
- Evaluation: Success rate of target class prediction

## Files

- `test.py` — Runs the attack and evaluates success rate
- `requirements.txt` — Required libraries

## How to Run

```bash
pip install -r requirements.txt
python test.py
```

## Example Output
```bash

```
## Notes
- For each image, a random target class (≠ true label) is generated.
- Attack is considered successful only if the model predicts the exact target class.
