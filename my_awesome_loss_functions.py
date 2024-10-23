import torch
import torch.nn as nn

# Enhanced loss with optional thresholding and clipping for MSE and MAE
class FlexibleThresholdedLoss(nn.Module):
    def __init__(self, use_mse_threshold=True, use_mae_threshold=True, mse_threshold=None, mae_threshold=None, 
                 reduction='mean', l1_weight=0.5, l2_weight=0.5, threshold_weight=0.5, non_threshold_weight=0.5,
                 mse_clip_ratio=None, mae_clip_ratio=None):
        """
        use_mse_threshold: Whether to apply a threshold to MSE-based loss.
        use_mae_threshold: Whether to apply a threshold to MAE-based loss.
        mse_threshold: Static MSE-based threshold if provided; otherwise, will use adaptive MSE threshold.
        mae_threshold: Static MAE-based threshold if provided; otherwise, will use adaptive MAE threshold.
        reduction: Specifies the reduction to apply to the output. ('mean' or 'sum').
        l1_weight: The weight for L1 loss in both thresholded and non-thresholded parts.
        l2_weight: The weight for L2 loss in both thresholded and non-thresholded parts.
        threshold_weight: Weight for the thresholded loss part.
        non_threshold_weight: Weight for the non-thresholded loss part.
        mse_clip_ratio: Clip the MSE-based thresholded loss when it exceeds this ratio of the current MSE threshold. If None, no clipping is applied.
        mae_clip_ratio: Clip the MAE-based thresholded loss when it exceeds this ratio of the current MAE threshold. If None, no clipping is applied.
        """
        super(FlexibleThresholdedLoss, self).__init__()
        self.use_mse_threshold = use_mse_threshold
        self.use_mae_threshold = use_mae_threshold
        self.mse_threshold = mse_threshold
        self.mae_threshold = mae_threshold
        self.reduction = reduction
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.threshold_weight = threshold_weight
        self.non_threshold_weight = non_threshold_weight
        self.mse_clip_ratio = mse_clip_ratio
        self.mae_clip_ratio = mae_clip_ratio

    def forward(self, input_img, target_img):
        # Calculate pixel-wise absolute difference
        pixel_diff = torch.abs(input_img - target_img)

        # Part 1: MSE-based threshold
        mse_loss = torch.mean((input_img - target_img) ** 2)  # MSE loss
        if self.use_mse_threshold:
            if self.mse_threshold is None:
                self.mse_threshold = mse_loss  # Set adaptive threshold based on MSE
            mse_thresholded_diff = torch.where(pixel_diff < self.mse_threshold, torch.tensor(0.0, device=pixel_diff.device), pixel_diff)
            mse_thresholded_l1_loss = mse_thresholded_diff  # L1 part for MSE-thresholded differences
            mse_thresholded_l2_loss = mse_thresholded_diff ** 2  # L2 part for MSE-thresholded differences

            # Apply clipping if mse_clip_ratio is provided
            if self.mse_clip_ratio is not None:
                max_mse_loss = self.mse_clip_ratio * self.mse_threshold
                mse_thresholded_l1_loss = torch.clamp(mse_thresholded_l1_loss, max=max_mse_loss)
                mse_thresholded_l2_loss = torch.clamp(mse_thresholded_l2_loss, max=max_mse_loss)
        else:
            mse_thresholded_l1_loss = pixel_diff
            mse_thresholded_l2_loss = pixel_diff ** 2

        # Part 2: MAE-based threshold
        mae_loss = torch.mean(pixel_diff)  # MAE loss
        if self.use_mae_threshold:
            if self.mae_threshold is None:
                self.mae_threshold = mae_loss  # Set adaptive threshold based on MAE
            mae_thresholded_diff = torch.where(pixel_diff < self.mae_threshold, torch.tensor(0.0, device=pixel_diff.device), pixel_diff)
            mae_thresholded_l1_loss = mae_thresholded_diff  # L1 part for MAE-thresholded differences
            mae_thresholded_l2_loss = mae_thresholded_diff ** 2  # L2 part for MAE-thresholded differences

            # Apply clipping if mae_clip_ratio is provided
            if self.mae_clip_ratio is not None:
                max_mae_loss = self.mae_clip_ratio * self.mae_threshold
                mae_thresholded_l1_loss = torch.clamp(mae_thresholded_l1_loss, max=max_mae_loss)
                mae_thresholded_l2_loss = torch.clamp(mae_thresholded_l2_loss, max=max_mae_loss)
        else:
            mae_thresholded_l1_loss = pixel_diff
            mae_thresholded_l2_loss = pixel_diff ** 2

        # Part 3: Non-thresholded loss (all differences are considered for both MSE and MAE)
        non_thresholded_l1_loss = pixel_diff  # L1 part without threshold
        non_thresholded_l2_loss = pixel_diff ** 2  # L2 part without threshold

        # Combine L1 and L2 losses for thresholded parts
        mse_thresholded_loss = self.l1_weight * mse_thresholded_l1_loss + self.l2_weight * mse_thresholded_l2_loss
        mae_thresholded_loss = self.l1_weight * mae_thresholded_l1_loss + self.l2_weight * mae_thresholded_l2_loss

        # Combine L1 and L2 losses for non-thresholded part
        non_thresholded_loss = self.l1_weight * non_thresholded_l1_loss + self.l2_weight * non_thresholded_l2_loss

        # Apply reduction (mean or sum) to each part
        if self.reduction == 'mean':
            mse_thresholded_loss = mse_thresholded_loss.mean()
            mae_thresholded_loss = mae_thresholded_loss.mean()
            non_thresholded_loss = non_thresholded_loss.mean()
        elif self.reduction == 'sum':
            mse_thresholded_loss = mse_thresholded_loss.sum()
            mae_thresholded_loss = mae_thresholded_loss.sum()
            non_thresholded_loss = non_thresholded_loss.sum()

        # Combine thresholded and non-thresholded losses with respective weights
        total_loss = self.threshold_weight * (mse_thresholded_loss + mae_thresholded_loss) + self.non_threshold_weight * non_thresholded_loss

        return total_loss

# Example usage
if __name__ == "__main__":
    # Create example images (generated and target) with random pixel values
    gen_img = torch.rand((1, 3, 256, 256))  # Generated image
    target_img = torch.rand((1, 3, 256, 256))  # Target image

    # Instantiate the flexible thresholded loss with separate clipping ratios for MSE and MAE
    loss_fn = FlexibleThresholdedLoss(use_mse_threshold=True, use_mae_threshold=True, reduction='mean', l1_weight=0.5, l2_weight=0.5, threshold_weight=0.7, non_threshold_weight=0.3, mse_clip_ratio=2.0, mae_clip_ratio=1.5)

    # Calculate the loss
    loss = loss_fn(gen_img, target_img)
    print(f"Flexible Thresholded and Clipped MSE & MAE Loss: {loss.item()}")
