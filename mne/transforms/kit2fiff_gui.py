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
from .transforms import apply_trans, coord_trans
from .viewer import HeadViewController, PointObject
from ..fiff.kit.coreg import read_hsp, read_elp, transform_ALS_to_RAS, \
                             get_neuromag_transform



use_editor = CheckListEditor(cols=5, values=[(i, str(i)) for i in xrange(5)])


class Kit2FiffPanel(HasTraits):
    """Control panel for kit2fiff conversion"""
    # Source Files
    sqd_file = File(filter=['*.sqd'])
    hsp_file = File(exists=True, filter=['*.pickled', '*.txt'],
                    desc="Digitizer head shape")
    fid_file = File(exists=True, filter=['*.txt'], desc="Digitizer fiducials")

    # Marker Points
    mrk_ALS = Array(float, shape=(5, 3))
    mrk = Property(Array(float, shape=(5, 3)), depends_on=('mrk_ALS'))
    use_mrk = List(range(5), desc="Which marker points to use for the device "
                   "head coregistration.")

    # Head Shape Fiducials
    elp_raw = Property(Array(float, shape=(8, 3)), depends_on=['fid_file'])
    neuromag_trans = Property(Array(float, shape=(3, 3)),
                              depends_on=['elp_raw'])
    elp_src = Property(Array(float, shape=(5, 3)), depends_on=['neuromag_trans'])
    fid_src = Property(Array(float, shape=(3, 3)), depends_on=['neuromag_trans'])
    dev_head_trans = Property(Array(float, shape=(4, 4)),
                              depends_on=['elp_src', 'use_mrk', 'mrk'])
    elp_dst = Property(Array(shape=(5, 3)), depends_on=['dev_head_trans'])
    fid_dst = Property(Array(shape=(3, 3)), depends_on=['dev_head_trans'])

    # Head Shape
    hsp_src = Property(Array(float, shape=(None, 3)), depends_on=['hsp_file'])  # hsp in neuromag
    hsp_dst = Property(Array(float, shape=(None, 3)), depends_on=['hsp_src', 'dev_head_trans'])  # hsp in neuromag

    # Events
    events = Array(Int, shape=(None,), value=[])
    endian = Enum("Little", "Big", desc="Binary coding of event channels")
    event_info = Property(Str, depends_on=['events', 'endian'])

    # Visualization
    scene = Instance(MlabSceneModel)
    fid_obj = Instance(PointObject)
    hsp_obj = Instance(PointObject)
#    sns_obj= Instance(PointObject)

    view = View(VGroup(VGroup(Item('sqd_file', label="Data"),
                              Item('hsp_file', label='Head Shape'),
                              Item('fid_file', label='Dig Points'),
                              Item('use_mrk', editor=use_editor, style='custom'),
                              label="Sources", show_border=True),
                       VGroup(Item('endian', style='custom'),
                              Item('event_info', style='readonly', show_label=False),
                              label='Events', show_border=True)))

    def _get_mrk(self):
        return transform_ALS_to_RAS(self.mrk_ALS, unit='m')

    def _get_elp_raw(self):
        if os.path.exists(self.fid_file):
            pts = read_elp(self.fid_file)
            return pts
        else:
            return np.empty((0, 3))

    def _get_neuromag_trans(self):
        if np.any(self.elp_raw):
            pts = transform_ALS_to_RAS(self.elp_raw[:3])
            nasion, lpa, rpa = pts
            trans = get_neuromag_transform(nasion, lpa, rpa)
            return trans
        else:
            return np.zeros((3, 3))

    def _get_fid_src(self):
        pts = self.elp_raw[:3]
        if np.any(pts):
            pts = transform_ALS_to_RAS(pts)
            pts = np.dot(pts, self.neuromag_trans.T)
        return pts

    def _get_elp_src(self):
        pts = self.elp_raw[3:]
        if np.any(pts):
            pts = transform_ALS_to_RAS(pts)
            pts = np.dot(pts, self.neuromag_trans.T)
        return pts

    def _get_dev_head_trans(self):
        if np.any(self.fid_src):
            src_pts = self.elp_src
        else:
            return None

        if np.any(self.mrk):
            dst_pts = self.mrk
        else:
            return None

        n_use = len(self.use_mrk)
        if n_use < 3:
            error(None, "Estimating the device head transform requires at "
                  "least 3 marker points. Please adjust the markers used.",
                  "Not Enough Marker Points")
            return
        elif n_use < 5:
            src_pts = src_pts[self.use_mrk]
            dst_pts = dst_pts[self.use_mrk]

        trans = fit_matched_pts(src_pts, dst_pts, params=False)
        return trans

    def _get_fid_dst(self):
        if np.any(self.dev_head_trans):
            return apply_trans(self.dev_head_trans, self.fid_src)
        else:
            return np.empty((0, 3))

    def _get_elp_dst(self):
        if np.any(self.dev_head_trans):
            return apply_trans(self.dev_head_trans, self.elp_src)
        else:
            return np.empty((0, 3))

    def _get_hsp_src(self):
        if os.path.exists(self.hsp_file):
            pts = read_hsp(self.hsp_file)
            pts = transform_ALS_to_RAS(pts)
            pts = np.dot(pts, self.neuromag_trans.T)
            return pts
        else:
            return np.empty((0, 3))

    def _get_hsp_dst(self):
        if np.any(self.dev_head_trans) and np.any(self.hsp_src):
            return apply_trans(self.dev_head_trans, self.hsp_src)
        else:
            return np.empty((0, 3))

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

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.fid_obj = PointObject(scene=self.scene, color=(25, 225, 25),
                                   point_scale=5e-3)
        self.sync_trait('fid_dst', self.fid_obj, 'points', mutual=False)

        self.elp_obj = PointObject(scene=self.scene, color=(50, 50, 220),
                                   point_scale=1e-2, opacity=.2)
        self.sync_trait('elp_dst', self.elp_obj, 'points', mutual=False)

        self.hsp_obj = PointObject(scene=self.scene, color=(200, 200, 200),
                                   point_scale=2e-3)
        self.sync_trait('hsp_dst', self.hsp_obj, 'points', mutual=False)



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

    @on_trait_change('scene.activated')
    def _init_plot(self):
        mrk_trans = coord_trans('ALS', 'RAS')
        self.marker_panel.mrk1_obj.trans = mrk_trans
        self.marker_panel.mrk2_obj.trans = mrk_trans
        self.marker_panel.mrk3_obj.trans = mrk_trans

        mrk = self.marker_panel.mrk3
        mrk.sync_trait('points', self.kit2fiff_panel, 'mrk_ALS', mutual=False)



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
