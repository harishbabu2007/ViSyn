import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as vutils
import wandb
from config import *
from generator import Generator
from discriminator import Discriminator
from landscape_dataset import LandscapesDataset
from losses import (
    VGGPerceptualLoss,
    generator_total_loss,
    discriminator_total_loss,
)


torch.manual_seed(SEED)

def denormalize(x):
    x = (x + 1.0) / 2.0
    return x.clamp(0, 1)


def save_checkpoint(
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    epoch,
    model_name,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "epoch": epoch,
    }

    path = os.path.join(
        CHECKPOINT_DIR,
        f"{model_name}_epoch_{epoch}.pth"
    )

    torch.save(checkpoint, path)

    print(f"checkpoint saved -> {path}")


# Sample Logging
@torch.no_grad()
def log_samples(
    generator,
    sample_condition,
    epoch,
    model_name,
):
    generator.eval()

    fake = generator(sample_condition)

    fake = denormalize(fake)

    condition = denormalize(sample_condition)

    if condition.shape[1] == 2:
        # seg + depth
        seg_vis = condition[:, 0:1]

        depth_vis = condition[:, 1:2]

        # create fake RGB visualization
        grid_condition = vutils.make_grid(
            torch.cat(
                [seg_vis, depth_vis, depth_vis],
                dim=1
            ),
            normalize=False,
        )
    elif condition.shape[1] == 1:
        # grayscale
        grid_condition = vutils.make_grid(
            condition.repeat(1, 3, 1, 1),
            normalize=False,
        )
    else:
        grid_condition = vutils.make_grid(
            condition[:, :3],
            normalize=False,
        )

    grid_fake = vutils.make_grid(
        fake,
        normalize=False,
    )

    wandb.log({
        f"{model_name}_condition": wandb.Image(grid_condition),
        f"{model_name}_generated": wandb.Image(grid_fake),
        "epoch": epoch,
    })

    generator.train()


# Training Loop
def train(
    generator,
    discriminator,
    loader,
    g_optimizer,
    d_optimizer,
    perceptual_loss_fn,
    model_type,
):
    generator.train()
    discriminator.train()
    

    for epoch in range(EPOCHS):
        sample_condition = None
        
        for batch_idx, batch in enumerate(loader):
            image = batch["image"].to(DEVICE)
            seg = batch["seg"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)

            if model_type == "sd2i":
                condition = torch.cat(
                    [seg, depth],
                    dim=1
                )

                real_target = image

            elif model_type == "s2d":
                condition = seg
                real_target = depth

            else:
                raise ValueError("invalid model type")

            # save one sample condition
            if sample_condition is None:
                sample_condition = condition[:1]

            fake_target = generator(condition)

            if model_type == "sd2i":

                real_input = torch.cat(
                    [seg, depth, image],
                    dim=1
                )

                fake_input = torch.cat(
                    [seg, depth, fake_target.detach()],
                    dim=1
                )

            else:

                real_input = torch.cat(
                    [seg, depth],
                    dim=1
                )

                fake_input = torch.cat(
                    [seg, fake_target.detach()],
                    dim=1
                )

            d_optimizer.zero_grad()

            real_preds = discriminator(real_input)

            fake_preds = discriminator(fake_input)

            d_losses = discriminator_total_loss(
                real_preds,
                fake_preds,
                real_input,
            )

            d_loss = d_losses["total"]

            # NaN protection
            if torch.isnan(d_loss):

                print("NaN detected in discriminator")

                continue

            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=1.0
            )

            d_optimizer.step()
            g_optimizer.zero_grad()

            if model_type == "sd2i":

                fake_input = torch.cat(
                    [seg, depth, fake_target],
                    dim=1
                )

            else:

                fake_input = torch.cat(
                    [seg, fake_target],
                    dim=1
                )

            fake_preds = discriminator(fake_input)

            g_losses = generator_total_loss(
                fake_preds,
                fake_target,
                real_target,
                perceptual_loss_fn,
            )

            g_loss = g_losses["total"]

            # NaN protection
            if torch.isnan(g_loss):
                print("NaN detected in generator")

                continue

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=1.0
            )

            g_optimizer.step()


            if batch_idx % 25 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(loader)}] "
                    f"G Loss: {g_loss.item():.4f} "
                    f"D Loss: {d_loss.item():.4f}"
                )

                wandb.log({
                    "g_total": g_loss.item(),
                    "g_adv": g_losses["adv"].item(),
                    "g_l1": g_losses["l1"].item(),
                    "g_perceptual": g_losses["perceptual"].item(),

                    "d_total": d_loss.item(),
                    "d_adv": d_losses["adv"].item(),
                    "d_r1": d_losses["r1"].item(),

                    "epoch": epoch,
                })

        if epoch % SAMPLE_EVERY == 0:
            log_samples(
                generator,
                sample_condition,
                epoch,
                model_type,
            )

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                epoch,
                model_type,
            )



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["s2d", "sd2i"]
    )

    args = parser.parse_args()

    model_type = args.model


    dataset = LandscapesDataset(
        num_images=NUM_IMAGES,
        input_dir="./dataset/inputs",
        segment_dir="./dataset/segments",
        depth_dir="./dataset/depths",
        image_size=(IMG_SIZE, IMG_SIZE),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if model_type == "sd2i":
        generator = Generator(
            condition_channels=2,
            output_channels=3,
        ).to(DEVICE)

        discriminator = Discriminator(
            in_channels=5
        ).to(DEVICE)

    else:
        generator = Generator(
            condition_channels=1,
            output_channels=1,
        ).to(DEVICE)

        discriminator = Discriminator(
            in_channels=2
        ).to(DEVICE)


    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
    )

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
    )


    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)

    # mixed precsion coz my gpu bad

    # WandB stuff
    wandb.init(
        project=WANDB_PROJECT,
        name=model_type,
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "model": model_type,
        }
    )

    train(
        generator,
        discriminator,
        loader,
        g_optimizer,
        d_optimizer,
        perceptual_loss_fn,
        model_type,
    )


if __name__ == "__main__":
    main()