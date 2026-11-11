
class MemoryController:
    def __init__(
        self,
        moderate_threshold=0.02,
        high_threshold=0.10,
        base_weight=0.5,
        moderate_weight=0.7,
        high_weight=0.9
    ):
        if not 0 <= moderate_threshold < high_threshold:
            raise ValueError(
                "Los umbrales deben ser válidos y crecientes"
            )

        weights = [
            base_weight,
            moderate_weight,
            high_weight
        ]

        if any(not 0 <= w <= 1 for w in weights):
            raise ValueError(
                "Los pesos deben estar entre 0 y 1"
            )

        if weights != sorted(weights):
            raise ValueError(
                "Los pesos deben ser crecientes"
            )

        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold

        self.base_weight = base_weight
        self.moderate_weight = moderate_weight
        self.high_weight = high_weight

    def calculate_weight(self, loss_increase):
        if loss_increase >= self.high_threshold:
            return self.high_weight

        if loss_increase >= self.moderate_threshold:
            return self.moderate_weight

        return self.base_weight
