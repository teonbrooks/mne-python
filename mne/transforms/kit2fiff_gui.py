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

from .marker_gui import MarkerPanel, MarkerPointDest
from .coreg import fit_matched_pts
from .viewer import HeadViewController, PointObject
from ..fiff.kit.coreg import read_hsp, write_hsp



use_editor = CheckListEditor(cols=5, values=[(i, str(i)) for i in xrange(5)])


class Kit2FiffPanel(HasTraits):
    """Control panel for kit2fiff conversion"""
    # Source Files
    sqd_file = File(filter=['*.sqd'])
    hsp_file = File(exists=True, filter=['*.pickled', '*.txt'],
                    desc="Digitizer head shape")
    fid_file = File(exists=True, filter=['*.txt'], desc="Digitizer fiducials")

    mrk = Array(float, shape=(5, 3))

    # Marker Points
    use_mrk = List(range(5), desc="Which marker points to use for the device "
                   "head coregistration.")

    hsp_src = Property(Array(float, shape=(None, 3)), depends_on=['hsp_file'])  # hsp in neuromag
    dev_head_trans = Property(Array(float, shape=(4, 4)), depends_on=['use_mrk', 'fid_file', 'mrk'])
    hsp_dst = Property(Array(float, shape=(None, 3)), depends_on=['hsp_src', 'dev_head_trans'])  # hsp in neuromag

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

    def _get_hsp_src(self):
        if os.path.exists(self.hsp_file):
            pts = read_hsp(self.hsp_file)
            return pts
        else:
            return np.empty((0, 3))
    
    def _get_dev_head_trans(self):
        if 
        
        trans = fit_matched_pts(src_pts, tgt_pts, params=False)


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
