from typing import Dict

"""
Implements Backdoor Attack , Random Label Corruption  
"""


class LabelCorruption:
    """ This is the Base Class for Image Corruptions. """
    def __init__(self, attack_config: Dict):
        self.attack_config = attack_config
        self.noise_model = self.attack_config.get("noise_model", None)
        self.frac_adv = self.attack_config.get('frac_adv', 0)

        self.target_class = self.attack_config.get('backdoor_label', 0)

        self.num_corrupt = 0
        self.curr_corr = 0

    def attack(self, X, Y):
        if self.curr_corr > 0:
            # apply attack
            for ix, sample in enumerate(Y):
                Y[ix] = self.corrupt()
        return X, Y

    def corrupt(self):
        raise NotImplementedError('You need to Implement this method for each attack class')


class Backdoor(LabelCorruption):
    def __init__(self, attack_config: Dict):
        LabelCorruption.__init__(self, attack_config=attack_config)

    def corrupt(self):
        return self.target_class
