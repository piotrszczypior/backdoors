# Backdoor Configuration Guide

This document describes how to configure the backdoor system using the JSON configuration files located in `config/backdoors/`.

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `poison_rate` | `float` | The fraction of the dataset to be poisoned (between 0.0 and 1.0). |
| `trigger_type` | `string` | The visual modification to apply to the image. |
| `attack_mode` | `string` | High-level attack type: `dirty_label` (label changed) or `clean_label` (label preserved). |
| `target_mapping` | `string` | How the labels of poisoned samples are transformed. |
| `selector_type` | `string` | Strategy for choosing which samples to poison. |
| `target_class` | `int` | The class ID that the poisoned samples should be classified as (for Dirty-Label attacks). |
| `source_classes` | `list[int]` | List of class IDs to target when using `source_selector`. |
| `seed` | `int` | Random seed for deterministic poisoning. |

---

## Available Components

### 1. Triggers (`trigger_type`)
*   `white_box`: Adds a 10x10 white square in the center of the image.
*   `gaussian_noise`: Adds Gaussian noise to the image.

### 2. Selectors (`selector_type`)
*   `random_selector`: Selects `poison_rate * total_samples` random indices from the entire dataset.
*   `source_selector`: Selects `poison_rate * samples_in_source_classes` from the specified `source_classes`.

### 3. Target Mappings (`target_mapping`)
*   `all_to_one`: Always maps poisoned samples to `target_class`.
*   `source_to_target`: Maps samples from `source_classes` to `target_class`.
*   `clean_label`: Keeps the original label (used for Clean-Label attacks).

---

## Example Scenarios

### Case 1: Dirty-Label All-to-One
Poisons random samples from any class and changes their label to the target class.
```json
{
    "poison_rate": 0.05,
    "trigger_type": "white_box",
    "attack_mode": "dirty_label",
    "target_mapping": "all_to_one",
    "selector_type": "random_selector",
    "target_class": 281
}
```

### Case 2: Dirty-Label Source-to-One
Poisons only samples from specific classes and changes their label to the target class.
```json
{
    "poison_rate": 0.05,
    "trigger_type": "white_box",
    "attack_mode": "dirty_label",
    "target_mapping": "source_to_target",
    "selector_type": "source_selector",
    "source_classes": [1, 2, 3, 4, 5],
    "target_class": 281
}
```

### Case 3: Clean-Label Attack
Poisons samples from specific classes but keeps their original labels.
```json
{
    "poison_rate": 0.05,
    "trigger_type": "white_box",
    "attack_mode": "clean_label",
    "target_mapping": "clean_label",
    "selector_type": "source_selector",
    "source_classes": [1, 2, 3, 4, 5],
    "target_class": 281
}
```
