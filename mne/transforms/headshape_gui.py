"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import cPickle as pickle
import os

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.mlab import text3d, pipeline
from mayavi.modules.glyph import Glyph
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import HasTraits, on_trait_change, Instance, Property, \
                       Array, Bool, Button, Color, Enum, File, Float, List, \
                       Range, Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, HSplit, CheckListEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import fit_matched_pts, decimate_headshape
from .marker_gui import ALSHeadViewController
from .transforms import apply_trans, rotation, translation
from ..fiff.kit.coreg import read_hsp



out_wildcard = ("Pickled head shape (*.pickled)|*.pickled|"
                "Hsp (text) file (*.hsp)|*.hsp")
out_ext = ['.pickled', '.hsp']



class PointObject(HasTraits):
    """Represent 5 marker points"""
    points = Array(float, shape=(None, 3))
    point_scale = Float(1., label='Point Scale')
    triangles = Array(int, shape=(None, 3))
    name = Str

#    style = Enum("Surface", "Wireframe", "Points")
    color = Color()
    rgbcolor = Property(Tuple(Float, Float, Float), depends_on='color')

    scene = Instance(MlabSceneModel, ())
    glyph = Instance(Glyph)
    src = Instance(VTKDataSource)

    def _get_rgbcolor(self):
        return tuple(v / 255. for v in self.color.Get())

#    @on_trait_change('scene.activated')
    def plot_points(self, color=None):
        if color is not None:
            self.color = tuple(int(c * 255) for c in color)

        fig = self.scene.mayavi_scene

        x, y, z = self.points.T
        status = np.ones(len(x))
        scatter = pipeline.scalar_scatter(x, y, z, status)
        glyph = pipeline.glyph(scatter, color=self.rgbcolor, figure=fig,
                               scale_factor=1e-2)  # self.point_scale)
        self.src = scatter
        self.glyph = glyph

        self.sync_trait('points', self.src.data, 'points', mutual=False)


class HeadShape(HasTraits):
    file = File(exists=True)
    hsp_points = Array(float, shape=(None, 3))
    points = Array(float, shape=(None, 3))
    resolution = Range(value=10, low=1, high=100)

    view = View(VGroup('file', 'resolution', label="Head Shape Source",
                       show_border=True))

    @on_trait_change('file')
    def load(self, fname):
        pts = read_hsp(fname, None)
        self.hsp_points = pts

    @on_trait_change('hsp_points,resolution')
    def update_points(self):
        pts = decimate_headshape(self.hsp_points, self.resolution)
        self.points = pts




class ControlPanel(HasTraits):
    """Has two marker points sources and interpolates to a third one"""
    file = File(exists=True)

    scene = Instance(MlabSceneModel, ())
    headview = Instance(ALSHeadViewController)
    headshape = Instance(HeadShape)
    headobj = Instance(PointObject)

    view = View(VGroup(Item('headshape', style='custom'),
                       Item('headview', style='custom'),
                       show_labels=False,
                       ))

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.headobj.plot_points(color=(.1, .9, 1))
        self.headshape.sync_trait('points', self.headobj.src.data, 'points',
                                  mutual=False)

#        self.headshape.on_trait_change(self.headobj.update, 'points')



class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to pre- and post measurement marker files (*.sqd) or empty string.
    """
    scene = Instance(MlabSceneModel, ())

    headview = Instance(ALSHeadViewController)
    headshape = Instance(HeadShape)
    headobj = Instance(PointObject)

    panel = Instance(ControlPanel)

    def _headview_default(self):
        return ALSHeadViewController(scene=self.scene)

    def _headshape_default(self):
        if os.path.exists(self._file):
            hs = HeadShape(scene=self.scene, file=self._file)
        else:
            hs = HeadShape(scene=self.scene)
        return hs

    def _headobj_default(self):
        ho = PointObject(scene=self.scene, points=self.headshape.points)
        return ho

    def _panel_default(self):
        return ControlPanel(scene=self.scene, headview=self.headview,
                            headshape=self.headshape, headobj=self.headobj)

    view = View(
#                HSplit
                HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
#                            dock='vertical'
                            ),
                       Item('panel', style="custom"),
                       show_labels=False,
                      ),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)

    def __init__(self, hsp=''):
        self._file = hsp
