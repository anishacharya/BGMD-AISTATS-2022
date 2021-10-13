# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from .grad_attack_models import *
from .image_corruption_models import *
from .backdoor import *


def get_grad_attack(attack_config: Dict):
    if attack_config["attack_model"] == 'drift':
        return DriftAttack(attack_config=attack_config)
    elif attack_config["attack_model"] == 'additive':
        return Additive(attack_config=attack_config)
    elif attack_config["attack_model"] == 'random':
        return Random(attack_config=attack_config)
    elif attack_config["attack_model"] == 'bit_flip':
        return BitFlipAttack(attack_config=attack_config)
    elif attack_config["attack_model"] == 'random_sign_flip':
        return RandomSignFlipAttack(attack_config=attack_config)
    else:
        return None


def get_feature_attack(attack_config: Dict):
    if attack_config["noise_model"] == 'additive':
        return ImageAdditive(attack_config=attack_config)
    elif attack_config["noise_model"] == 'impulse':
        return ImageImpulse(attack_config=attack_config)
    elif attack_config["noise_model"] == 'backdoor':
        return Backdoor(attack_config=attack_config)
    else:
        return None
