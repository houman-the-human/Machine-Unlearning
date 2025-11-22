import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler


class ConditionalDDPM(nn.Module):
    """
    A clean DDPM wrapper around:
      - UNet2DConditionModel
      - Class embedding
      - DDPMScheduler

    No training loop here. Only model definition + utility functions.
    """

    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels,
        down_block_types,
        up_block_types,
        cross_attention_dim: int,
        num_train_timesteps: int,
        beta_schedule: str,
        num_classes: int,
        embedding_dim: int,
    ):
        super().__init__()

        # --- Conditional UNet ---
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=cross_attention_dim,
        )

        # --- Class embedding ---
        self.class_embedder = nn.Embedding(num_classes, embedding_dim)

        # --- Diffusion scheduler ---
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
        )

    # -----------------------------------------------------------
    # Forward pass used during training
    # -----------------------------------------------------------
    def forward(self, images, timesteps, labels):
        """
        images:      (B, C, H, W)
        timesteps:   (B,)
        labels:      (B,) class indices
        """
        embeddings = self.class_embedder(labels).unsqueeze(1)  # (B, 1, emb_dim)

        noise_pred = self.unet(
            images,
            timesteps,
            encoder_hidden_states=embeddings
        ).sample

        return noise_pred

    # -----------------------------------------------------------
    # Sampling function (single-step, iterative loop done outside)
    # -----------------------------------------------------------
    @torch.no_grad()
    def predict_noise(self, x_t, t, labels):
        """
        Utility for sampling: predicts noise using UNet.
        """
        embeddings = self.class_embedder(labels).unsqueeze(1)
        return self.unet(x_t, t, encoder_hidden_states=embeddings).sample

    def add_noise(self, images, noise, timesteps):
        """
        Wrapper for scheduler.add_noise()
        """
        return self.noise_scheduler.add_noise(images, noise, timesteps)

    # -----------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------
    def save(self, unet_path, embedder_path):
        torch.save(self.unet.state_dict(), unet_path)
        torch.save(self.class_embedder.state_dict(), embedder_path)

    def load(self, unet_path, embedder_path, device):
        self.unet.load_state_dict(torch.load(unet_path, map_location=device))
        self.class_embedder.load_state_dict(torch.load(embedder_path, map_location=device))

