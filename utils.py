import argparse
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Compute Dice Loss for multi-class segmentation.
        
        Args:
        - pred (Tensor): Predicted output of shape (B, C, H, W) or (B, C, D, H, W)
        - target (Tensor): Ground truth of same shape (one-hot encoded)

        Returns:
        - Dice Loss (Tensor): Scalar value
        """
        assert pred.shape == target.shape, "Pred and target must have the same shape"

        # Flatten all except the batch and channel dimensions
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=2)
        denominator = pred.sum(dim=2) + target.sum(dim=2)

        # Compute Dice score per class and take the mean over channels
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss, dice
