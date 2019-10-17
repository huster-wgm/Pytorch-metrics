# Pytorch-metrics

This is a repo. for evaluation metrics using Pytorch. 
The metrics.py is designed for evaluation tasks using two pytorch tensors as input. 
All implemented metric is compatible with any batch_size and devices(CPU or GPU).

```
y_pred << 4D tensor in [batch_size, channels, img_rows, img_cols]
y_true << 4D tensor in [batch_size, channels, img_rows, img_cols]

metric = MSE()
acc = metric(y_pred, y_true).item()
print("{} ==> {}".format(repr(metric), acc))
```

## Requirement
- python3
- pytorch >= 1.
- torchvision >= 0.2.0

## Implementation

- Image similarity
  * AE (Average Angular Error)
  * MSE (Mean Square Error)
  * PSNR (Peak Signal-to-Noise Ratio)
  * SSIM (Structural Similarity)
  * LPIPS (Learned Perceptual Image Patch Similarity)
  
- Accuray
  * OA(Overall Accuracy)
  * Precision
  * Recall
  * F1-score
  * Kapp coefficiency
  * Jaccard Index

## Ongoing
- FID(Fréchet Inception Distance)

## Acknowledgment
Our implementations are largely inspired by many open-sources codes, repos, as well as papers.
Many thanks to the authors.
* Richard Zhang, LPIPS(https://github.com/richzhang/PerceptualSimilarity)
* Jorge Pessoa, SSIM(https://github.com/jorge-pessoa/pytorch-msssim)

## LICENSE
This implementation is licensed under the MIT License.

