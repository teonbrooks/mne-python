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
                       Array, Bool, Button, Color, File, Float, Int, List, \
                       Range, Str, Tuple
from traitsui.api import View, Item, HGroup, VGroup, CheckListEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import decimate_headshape
from .viewer import HeadViewController
from ..fiff.kit.coreg import read_hsp, write_hsp



out_wildcard = ("Pickled head shape (*.pickled)|*.pickled|"
                "Text file (*.txt)|*.txt")
out_ext = ['.pickled', '.txt']



class PointObject(HasTraits):
    """Represent 5 marker points"""
    points = Array(float, shape=(None, 3))
    name = Str

    color = Color()
    rgbcolor = Property(Tuple(Float, Float, Float), depends_on='color')
    point_scale = Float(10, label='Point Scale')
    visible = Bool(True)

    scene = Instance(MlabSceneModel, ())

    view = View(HGroup(Item('point_scale', label='Point size'), 'color',
                       'visible'))

#    view2 = View(HGroup('visible', 'color',
#                        label="Reference Head Shape",
#                        show_border=True))

    def _get_rgbcolor(self):
        return tuple(v / 255. for v in self.color.Get())

    @on_trait_change('scene.activated')
    def plot_points(self):
        _scale = self.scene.camera.parallel_scale

        if hasattr(self, 'glyph'):
            self.glyph.remove()
        if hasattr(self, 'src'):
            self.src.remove()

        fig = self.scene.mayavi_scene

        self.info = self.points
        x, y, z = self.points.T
        scatter = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(scatter, color=self.rgbcolor, figure=fig,
                               scale_factor=self.point_scale, opacity=1.)
        self.src = scatter
        self.glyph = glyph

        self.sync_trait('points', self.src.data, 'points', mutual=False)
        self.sync_trait('point_scale', self.glyph.glyph.glyph, 'scale_factor')
        self.sync_trait('rgbcolor', self.glyph.actor.property, 'color', mutual=False)
        self.sync_trait('visible', self.glyph, 'visible')

        self.scene.camera.parallel_scale = _scale



class HeadShape(HasTraits):
    file = File(exists=True)
    hsp_points = Array(float, shape=(None, 3))
    points = Array(float, shape=(None, 3))
    ref_points = Array(float, shape=(None, 3))
    exclude = List(Int)
    n_points = Property(Int, depends_on='points')
    n_points_all = Int(0)
    resolution = Range(value=35, low=5, high=50, label="Resolution [mm]")

    save_as = Button(label="Save As...")

    view = View(VGroup('file', 'resolution',
                       Item('exclude', editor=CheckListEditor(), style='text'),
                       '_',
                       Item('n_points', label='N Points', style='readonly'),
                       Item('save_as', show_label=False),
                       label="Head Shape Source", show_border=True))

    def _get_n_points(self):
        return len(self.points)

    @on_trait_change('file')
    def load(self, fname):
        pts = read_hsp(fname, None)
        self._cache = {}
        self._exclude = {}
        self.hsp_points = pts

    def _exclude_changed(self, old, new):
        """Validate the values of the exclude list"""
        if not hasattr(self, '_cache'):
            return

        items = set(self.exclude)

        for i in sorted(items):
            if i > self.n_points_all or i < 0:
                items.remove(i)

        exclude = sorted(items)
        self._exclude[self.resolution] = exclude
        self.exclude = exclude
        self.update_points()

    def _exclude_items_changed(self, old, new):
        """This hack is necessary to update the editor for exclude"""
        items = list(self.exclude)
        if items:
            items.append(items[0])
        else:
            items.append(-1)
        self.exclude = items

    @on_trait_change('hsp_points')
    def update_ref_points(self):
        self.ref_points = self.get_dec_points(10)

    @on_trait_change('hsp_points,resolution')
    def update_base_points(self):
        if not hasattr(self, '_cache'):
            return

        res = self.resolution
        self.exclude = self._exclude.get(res, [])
        self.update_points()

    def update_points(self):
        res = self.resolution
        pts = self.get_dec_points(res)
        self.n_points_all = len(pts)
        if self.exclude:
            sel = np.ones(len(pts), dtype=bool)
            sel[self.exclude] = False
            pts = pts[sel]

        self.points = pts

    def get_dec_points(self, res):
        if not hasattr(self, '_cache'):
            raise RuntimeError("No hsp file loaded")

        if res in self._cache:
            pts = self._cache[res]
        else:
            pts = decimate_headshape(self.hsp_points, res)
            self._cache[res] = pts

        return pts

    def cache(self, resolutions=xrange(30, 40)):
        for res in resolutions:
            self.get_dec_points(res)

    def _save_as_fired(self):
        dlg = FileDialog(action="save as", wildcard=out_wildcard,
                         default_path=self.file)
        dlg.open()
        if dlg.return_code != OK:
            return

        ext = out_ext[dlg.wildcard_index]
        path = dlg.path
        if not path.endswith(ext):
            path = path + ext
            if os.path.exists(path):
                msg = ("The file %r already exists. Should it be replaced"
                       "?" % path)
                answer = confirm(None, msg, "Overwrite File?")
                if answer != YES:
                    return

        if ext == '.pickled':
            pts = np.asarray(self.points)
            with open(path, 'w') as fid:
                pickle.dump(pts, fid)
        elif ext == '.txt':
            write_hsp(path, self.points)
        else:
            error(None, "Not Implemented: %r" % ext)



class ControlPanel(HasTraits):
    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    headshape = Instance(HeadShape)
    headobj = Instance(PointObject)
    headobj_ref = Instance(PointObject)

    view = View(VGroup(Item('headshape', style='custom'),
                       VGroup(Item('headobj', show_label=False, style='custom'),
                              label='Decimated Head Shape',
                              show_border=True),
                       VGroup(Item('headobj_ref', show_label=False,
                                   style='custom'),
                              label='Reference Head Shape',
                              show_border=True),
                       Item('headview', style='custom'),
                       show_labels=False,
                       ))

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.headshape.sync_trait('points', self.headobj, 'points')
        self.headshape.sync_trait('ref_points', self.headobj_ref, 'points')

        fig = self.scene.mayavi_scene
        self.picker = fig.on_mouse_pick(self.picker_callback)
        self.picker.tolerance = 0.001

    @on_trait_change('headshape.file')
    def _on_file_changes(self):
        if self.headview:
            self.headview.left = True

    def picker_callback(self, picker):
        mygl = self.headobj.glyph
        if picker.actor not in mygl.actor.actors:
            return

        n = len(mygl.glyph.glyph_source.glyph_source.output.points)
        point_id = picker.point_id / n

        # If the no points have been selected, we have '-1'
        if point_id == -1:
            return

        idx = point_id
        for e_idx in sorted(self.headshape.exclude):
            if idx >= e_idx:
                idx += 1

        self.headshape.exclude.append(idx)



class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files"""
    scene = Instance(MlabSceneModel, ())

    headshape = Instance(HeadShape)
    headobj = Instance(PointObject)
    headobj_ref = Instance(PointObject)
    headview = Instance(HeadViewController)

    panel = Instance(ControlPanel)

    def _headshape_default(self):
        hs = HeadShape(scene=self.scene)
        return hs

    def _headobj_default(self):
        color = tuple(int(c * 255) for c in (.1, .9, 1))
        ho = PointObject(scene=self.scene, points=self.headshape.points,
                         color=color, point_scale=5)
        return ho

    def _headobj_ref_default(self):
        color = tuple(int(c * 255) for c in (.9, .9, .9))
        ho = PointObject(scene=self.scene, points=self.headshape.ref_points,
                         color=color, point_scale=2)
        return ho

    def _headview_default(self):
        hv = HeadViewController(scene=self.scene, scale=160, system='ARI')
        return hv

    def _panel_default(self):
        return ControlPanel(scene=self.scene, headview=self.headview,
                            headshape=self.headshape, headobj=self.headobj,
                            headobj_ref=self.headobj_ref)

    view = View(HGroup(Item('scene',
                            editor=SceneEditor(scene_class=MayaviScene)),
                       Item('panel', style="custom"),
                       show_labels=False,
                      ),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)
