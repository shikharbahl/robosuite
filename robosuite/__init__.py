import os

from robosuite.environments.base import make
from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole

from robosuite.environments.wipe_force import SawyerWipeForceEnv, PandaWipeForceEnv
from robosuite.environments.wipe_tactile import SawyerWipeTactileEnv, PandaWipeTactileEnv
from robosuite.environments.wipe_3d_tactile import SawyerWipe3DTactileEnv, PandaWipe3DTactileEnv
from robosuite.environments.wipe_pegs import SawyerWipePegsEnv, PandaWipePegsEnv
from robosuite.environments.door import SawyerDoorEnv, PandaDoorEnv
from robosuite.environments.free_space_traj import SawyerFreeSpaceTrajEnv, PandaFreeSpaceTrajEnv

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
