import random
import json
from typing import Dict, Any

class Architecture:
    def __init__(
        self,
        arch_type: str,
        pipeline_stages: int,
        cache_type: str,
        cache_size: int,
        compute_units: int,
        branch_predictor: str,
        out_of_order: bool,
        # нові фізичні параметри
        transistor_count: int,       # у мільйонах
        die_area_mm2: float,         # площа кристала в мм²
        supply_voltage: float,       # V
        max_frequency_ghz: float,    # ГГц
        cooling_type: str            # "air","liquid","passive"
    ):
        # логічні параметри
        self.arch_type = arch_type
        self.pipeline_stages = pipeline_stages
        self.cache_type = cache_type
        self.cache_size = cache_size
        self.compute_units = compute_units
        self.branch_predictor = branch_predictor
        self.out_of_order = out_of_order

        # фізичні параметри
        self.transistor_count = transistor_count
        self.die_area_mm2     = die_area_mm2
        self.supply_voltage   = supply_voltage
        self.max_frequency_ghz = max_frequency_ghz
        self.cooling_type     = cooling_type

    @staticmethod
    def random_architecture() -> 'Architecture':
        """Генерує випадкову архітектуру з розширеними фізичними параметрами."""
        return Architecture(
            arch_type=random.choice(["CPU","GPU","DSP","FPGA","ASIC"]),
            pipeline_stages=random.randint(4, 20),
            cache_type=random.choice(["Inclusive","Exclusive","Non-inclusive"]),
            cache_size=random.choice([1,2,4,8,16]),
            compute_units=random.randint(1, 64),
            branch_predictor=random.choice(["Static","Dynamic"]),
            out_of_order=random.choice([True, False]),
            transistor_count=random.randint(100, 2000),    # млн транзисторів
            die_area_mm2=random.uniform(50.0, 400.0),      # мм²
            supply_voltage=random.choice([0.9, 1.0, 1.1, 1.2]),  # В
            max_frequency_ghz=random.uniform(1.0, 4.0),    # ГГц
            cooling_type=random.choice(["air","liquid","passive"])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Повертає всі параметри у вигляді словника для JSON/виводу."""
        return {
            "arch_type":         self.arch_type,
            "pipeline_stages":   self.pipeline_stages,
            "cache_type":        self.cache_type,
            "cache_size":        self.cache_size,
            "compute_units":     self.compute_units,
            "branch_predictor":  self.branch_predictor,
            "out_of_order":      self.out_of_order,
            "transistor_count":  self.transistor_count,
            "die_area_mm2":      self.die_area_mm2,
            "supply_voltage":    self.supply_voltage,
            "max_frequency_ghz": self.max_frequency_ghz,
            "cooling_type":      self.cooling_type
        }

    def __str__(self) -> str:
        """Красивий вивід у консоль або лог."""
        return json.dumps(self.to_dict(), indent=2)
