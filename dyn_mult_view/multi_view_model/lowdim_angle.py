from appearance_flow_model import AppearanceFlowModel
from dyn_mult_view.mv3d.utils.tf_utils import *
import tensorflow as tf

class AppFlowLowDimAngle(AppearanceFlowModel):

    def decodeAngle(self):
        return lrelu(linear_msra(self.disp, 10, "a0"))
