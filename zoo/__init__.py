from importlib import import_module

from .lora import LoRA
from .full import Full
from .full_mom import Full_Mom


def _resolve(module_name, candidates):
    m = import_module(f"{__name__}.{module_name}")
    for c in candidates:
        if hasattr(m, c):
            return getattr(m, c)
    available = [k for k in dir(m) if k[:1].isupper()]
    raise ImportError(
        f"Cannot resolve class in zoo.{module_name}. "
        f"Tried={candidates}, available={available}"
    )


LoRA_FixedB = _resolve("lora_fixedB", ["LoRA_FixedB", "Lora_FixedB", "FedNCF_Lora_FixedB"])
LoRA_FixedA = _resolve("lora_fixedA", ["LoRA_FixedA", "Lora_FixedA", "FedNCF_Lora_FixedA"])

LoRA_MomA = _resolve("lora_momA", ["LoRA_MomA", "Lora_MomA"])
LoRA_MomB = _resolve("lora_momB", ["LoRA_MomB", "Lora_MomB"])
LoRA_MomAB = _resolve("lora_momAB", ["LoRA_MomAB", "Lora_MomAB"])

LoRA_MomA_FixedB_Oneway = _resolve(
    "lora_momA_fixedB_oneway",
    ["LoRA_MomA_FixedB_Oneway", "Lora_MomA_FixedB_Oneway"]
)
LoRA_MomB_FixedA = _resolve("lora_momB_fixedA", ["LoRA_MomB_FixedA", "Lora_MomB_FixedA"])
LoRA_MomA_FixedB = _resolve("lora_momA_fixedB", ["LoRA_MomA_FixedB", "Lora_MomA_FixedB"])

Analyze_Full = _resolve("analyze_full", ["Analyze_Full"])
Analyze_LoRA = _resolve("analyze_lora", ["Analyze_LoRA", "Analyze_Lora"])
Analyze_LoRA_FixedB = _resolve("analyze_lora_fixedB", ["Analyze_LoRA_FixedB", "Analyze_Lora_FixedB"])
Analyze_LoRA_MomA_FixedB = _resolve(
    "analyze_lora_momA_fixedB",
    ["Analyze_LoRA_MomA_FixedB", "Analyze_Lora_MomA_FixedB"]
)
