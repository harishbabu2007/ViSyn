import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from config import *



# Hinge GAN Loss

def discriminator_hinge_loss(real_preds, fake_preds):

    real_loss = torch.mean(
        F.relu(1.0 - real_preds)
    )

    fake_loss = torch.mean(
        F.relu(1.0 + fake_preds)
    )

    return real_loss + fake_loss


def generator_hinge_loss(fake_preds):

    return -torch.mean(fake_preds)



# R1 Regularization

def r1_regularization(
    real_preds,
    real_images,
):

    gradients = torch.autograd.grad(
        outputs=real_preds.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]

    gradients = gradients.view(
        gradients.shape[0],
        -1
    )

    gradient_penalty = gradients.pow(2).sum(dim=1).mean()

    return gradient_penalty


# VGG Perceptual Loss

class VGGPerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1
        ).features

        # use first layers only
        self.feature_extractor = nn.Sequential(
            *list(vgg.children())[:35]
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()

    def forward(
        self,
        fake,
        real,
    ):

        # depth maps are 1-channel
        # convert -> 3 channel

        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)

        if real.shape[1] == 1:
            real = real.repeat(1, 3, 1, 1)

        # VGG expects [0,1]
        # current range is [-1,1]

        fake = (fake + 1.0) / 2.0
        real = (real + 1.0) / 2.0

        fake_features = self.feature_extractor(fake)

        real_features = self.feature_extractor(real)

        loss = F.l1_loss(
            fake_features,
            real_features
        )

        return loss



# Reconstruction Loss

def reconstruction_loss(fake, real):

    return F.l1_loss(fake, real)


# Full Generator Loss

def generator_total_loss(
    fake_preds,
    fake_images,
    real_images,
    perceptual_loss_fn,
):

    # adversarial
    adv_loss = generator_hinge_loss(fake_preds)

    # reconstruction
    l1_loss = reconstruction_loss(
        fake_images,
        real_images
    )

    # perceptual
    perceptual_loss = perceptual_loss_fn(
        fake_images,
        real_images
    )

    total_loss = (
        adv_loss
        + (LAMBDA_L1 * l1_loss)
        + (LAMBDA_PERCEPTUAL * perceptual_loss)
    )

    return {
        "total": total_loss,
        "adv": adv_loss,
        "l1": l1_loss,
        "perceptual": perceptual_loss,
    }



# Full Discriminator Loss

def discriminator_total_loss(
    real_preds,
    fake_preds,
    real_inputs,
):

    # hinge
    adv_loss = discriminator_hinge_loss(
        real_preds,
        fake_preds,
    )

    # R1
    r1_loss = torch.tensor(
        0.0,
        device=real_inputs.device
    )

    total_loss = adv_loss + (LAMBDA_R1 * r1_loss)

    return {
        "total": total_loss,
        "adv": adv_loss,
        "r1": r1_loss,
    }