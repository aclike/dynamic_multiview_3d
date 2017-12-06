from appearance_flow_model import AppearanceFlowModel
from dyn_mult_view.mv3d.utils.tf_utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

class AppFlowHighDimAngle(AppearanceFlowModel):

    def decodeAngle(self):
	a0 = lrelu(linear_msra(self.disp, 19, "a0"))
	a1 = lrelu(linear_msra(self.disp, 128, "a1"))
	return lrelu(linear_msra(self.disp, 256, "a2"))
