import torch
import torch.nn as nn

# Enhanced loss with separate handling for L1 and L2 (MSE) losses with optional thresholding
class FlexibleThresholdedLoss(nn.Module):
    def __init__(self, use_mse_threshold=True, use_mae_threshold=True, mse_threshold=None, mae_threshold=None, 
                 reduction='mean', l1_weight=0.5, l2_weight=0.5, threshold_weight=0.5, non_threshold_weight=0.5):
        """
        use_mse_threshold: Whether to apply a threshold to L2 (MSE)-based loss.
        use_mae_threshold: Whether to apply a threshold to L1-based loss.
        mse_threshold: Static L2 (MSE)-based threshold if provided; otherwise, will use adaptive MSE threshold.
        mae_threshold: Static L1-based threshold if provided; otherwise, will use adaptive MAE threshold.
        reduction: Specifies the reduction to apply to the output. ('mean' or 'sum').
        l1_weight: The weight for L1 loss in both thresholded and non-thresholded parts.
        l2_weight: The weight for L2 (MSE) loss in both thresholded and non-thresholded parts.
        threshold_weight: Weight for the thresholded loss part.
        non_threshold_weight: Weight for the non-thresholded loss part.
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

    def forward(self, input_img, target_img):
        # Calculate pixel-wise absolute difference (for L1) and squared difference (for L2/MSE)
        pixel_diff = torch.abs(input_img - target_img)  # For L1
        pixel_diff_squared = (input_img - target_img) ** 2  # For L2 (MSE)

        # Part 1: L2 (MSE)-based threshold handling
        mse_loss = pixel_diff_squared.mean()  # MSE (L2) loss
        if self.use_mse_threshold:
            if self.mse_threshold is None:
                self.mse_threshold = mse_loss  # Set adaptive threshold based on MSE
            mse_thresholded_diff = torch.where(pixel_diff_squared < self.mse_threshold, torch.tensor(0.0, device=pixel_diff.device), pixel_diff_squared)
            mse_thresholded_loss = mse_thresholded_diff  # L2 (MSE) part for pixels exceeding MSE threshold
        else:
            # No thresholding, use all squared differences for L2 (MSE)
            mse_thresholded_loss = pixel_diff_squared

        # Part 2: L1-based threshold handling
        mae_loss = pixel_diff.mean()  # L1 (absolute difference) loss
        if self.use_mae_threshold:
            if self.mae_threshold is None:
                self.mae_threshold = mae_loss  # Set adaptive threshold based on MAE
            mae_thresholded_diff = torch.where(pixel_diff < self.mae_threshold, torch.tensor(0.0, device=pixel_diff.device), pixel_diff)
            mae_thresholded_loss = mae_thresholded_diff  # L1 part for pixels exceeding MAE threshold
        else:
            # No thresholding, use all absolute differences for L1
            mae_thresholded_loss = pixel_diff

        # Part 3: Non-thresholded loss (all differences are considered for both L1 and L2)
        non_thresholded_l1_loss = pixel_diff  # L1 part without threshold
        non_thresholded_l2_loss = pixel_diff_squared  # L2 (MSE) part without threshold

        # Combine thresholded L1 and L2 losses
        combined_thresholded_loss = self.l1_weight * mae_thresholded_loss + self.l2_weight * mse_thresholded_loss

        # Combine non-thresholded L1 and L2 losses
        combined_non_thresholded_loss = self.l1_weight * non_thresholded_l1_loss + self.l2_weight * non_thresholded_l2_loss

        # Apply reduction (mean or sum) to each part
        if self.reduction == 'mean':
            combined_thresholded_loss = combined_thresholded_loss.mean()
            combined_non_thresholded_loss = combined_non_thresholded_loss.mean()
        elif self.reduction == 'sum':
            combined_thresholded_loss = combined_thresholded_loss.sum()
            combined_non_thresholded_loss = combined_non_thresholded_loss.sum()

        # Combine thresholded and non-thresholded losses with respective weights
        total_loss = self.threshold_weight * combined_thresholded_loss + self.non_threshold_weight * combined_non_thresholded_loss

        return total_loss

# Example usage
if __name__ == "__main__":
    # Create example images (generated and target) with random pixel values
    gen_img = torch.rand((1, 3, 256, 256))  # Generated image
    target_img = torch.rand((1, 3, 256, 256))  # Target image

    # Instantiate the flexible thresholded loss with options to enable or disable thresholds
    loss_fn = FlexibleThresholdedLoss(use_mse_threshold=True, use_mae_threshold=False, reduction='mean', l1_weight=0.5, l2_weight=0.5, threshold_weight=0.7, non_threshold_weight=0.3)

    # Calculate the loss
    loss = loss_fn(gen_img, target_img)
    print(f"Flexible Thresholded MSE & MAE Loss: {loss.item()}")
