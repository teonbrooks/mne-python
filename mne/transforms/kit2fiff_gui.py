"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import cPickle as pickle
import os

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.mlab import pipeline
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import HasTraits, on_trait_change, Instance, Property, \
                       Array, Bool, Button, Color, Enum, File, Float, Int, List, \
                       Range, Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, CheckListEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from .marker_gui import use_editor
from .marker_gui import MarkerPanel
from .viewer import HeadViewController
from ..fiff.kit.coreg import read_hsp, write_hsp


class Kit2FiffPanel(HasTraits):
    """Control panel for kit2fiff conversion"""
    # Source Files
    sqd = File(exists=True, filter=['*.sqd'])
    hsp = File(exists=True, filter=['*.pickled', '*.txt'], desc="Digitizer "
               "head shape")
    fid = File(exists=True, filter=['*.txt'], desc="Digitizer fiducials")

    # Marker Points
    use_mrk = List(range(5), desc="Which marker points to use for the device "
                   "head coregistration.")

    # Events
    events = Array(Int, shape=(None,), value=[])
    endian = Enum("Little", "Big", desc="Binary coding of event channels")
    event_info = Property(Str, depends_on=['events', 'endian'])

    view = View(VGroup(VGroup(Item('sqd', label="Data"),
                              Item('hsp', label='Head Shape'),
                              Item('fid', label='Dig Points'),
                              Item('use_mrk', editor=use_editor, style='custom'),
                              label="Sources", show_border=True),
                       VGroup(Item('endian', style='custom'),
                              Item('event_info', style='readonly', show_label=False),
                              label='Events', show_border=True)))

    def _get_event_info(self):
        """
        Return a string with the number of events found for each trigger value
        """
        if len(self.events) == 0:
            return "No events found."

        count = ["Events found:"]
        events = np.array(self.events)
        for i in np.unique(events):
            n = np.sum(events == i)
            count.append('%3i: %i' % (i, n))

        return os.linesep.join(count)


class ControlPanel(HasTraits):
    scene = Instance(MlabSceneModel, ())
    marker_panel = Instance(MarkerPanel)
    kit2fiff_panel = Instance(Kit2FiffPanel)

    view = View(Group(Item('marker_panel', label="Markers", style="custom",
                           dock='tab'),
                      Item('kit2fiff_panel', label="Kit2Fiff", style="custom",
                           dock='tab'),
                      layout='tabbed', show_labels=False)
                      )

    def _marker_panel_default(self):
        panel = MarkerPanel(scene=self.scene)
        return panel

    def _kit2fiff_panel_default(self):
        panel = Kit2FiffPanel(scene=self.scene)
        return panel



class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files"""
    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    control = Instance(ControlPanel)

    def _headview_default(self):
        hv = HeadViewController(scene=self.scene, scale=160, system='RAS')
        return hv

    def _control_default(self):
        p = ControlPanel(scene=self.scene)
        return p

    view = View(HGroup(Item('scene',
                            editor=SceneEditor(scene_class=MayaviScene)),
                       VGroup(Item('headview', style='custom'),
                              Item('control', style='custom'),
                              show_labels=False),
                       show_labels=False,
                      ),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)
