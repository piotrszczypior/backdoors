import torchvision.models as models


def _build(model_fn, weights):
    return model_fn(weights=weights)


def get_resnet152():
    return _build(
        models.resnet152,
        models.ResNet152_Weights.IMAGENET1K_V1,
    )


def get_efficientnet_b4():
    return _build(
        models.efficientnet_b4,
        models.EfficientNet_B4_Weights.IMAGENET1K_V1,
    )


def get_vit_b_16():
    return _build(
        models.vit_b_16,
        models.ViT_B_16_Weights.IMAGENET1K_V1,
    )


def get_deit_base():
    return _build(
        models.deit_base_patch16_224,
        models.DeiT_Base_Patch16_224_Weights.IMAGENET1K_V1,
    )
