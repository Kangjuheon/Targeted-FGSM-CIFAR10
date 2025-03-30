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
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 23.98it/s, loss=0.499]
[Epoch 1] Avg Loss: 0.5771
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 24.01it/s, loss=0.13] 
[Epoch 2] Avg Loss: 0.3799
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 23.97it/s, loss=0.358] 
[Epoch 3] Avg Loss: 0.3493
Epoch 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 24.05it/s, loss=0.325] 
[Epoch 4] Avg Loss: 0.3293
Epoch 5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 23.97it/s, loss=0.504] 
[Epoch 5] Avg Loss: 0.3178
Clean Evaluation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:10<00:00,  3.84it/s] 

[Clean Accuracy] 90.05%
FGSM Targeted Evaluation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:30<00:00,  1.32it/s] 

[FGSM Targeted Attack Success Rate] eps=0.03 → 28.47%
```
## Notes
- For each image, a random target class (≠ true label) is generated.
- Attack is considered successful only if the model predicts the exact target class.
