import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "../../../assets")


from .robotiq3 import getHandModel as getRobotiq3HandModel
from .panda import getHandModel as getPandaHandModel
from .allegro import getHandModel as getAllegroHandModel
from .ability_hand import getHandModel as getAbilityHandModel
from .shadow import getHandModel as getShadowHandModel
from .robotiq2 import getHandModel as getRobotiq2HandModel

_REGISTRY = {
    "robotiq3": getRobotiq3HandModel,
    "panda": getPandaHandModel,
    "allegro": getAllegroHandModel,
    "ability_hand": getAbilityHandModel,
    "shadow_hand": getShadowHandModel,
    "robotiq2": getRobotiq2HandModel,
}

AVAILABLE_HANDS = list(_REGISTRY.keys())


def get_hand_model(hand_name: str, device: str, **kwargs):
    return _REGISTRY[hand_name](device, ASSET_DIR, **kwargs)
