"""Mayavi/traits GUI elements"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from mayavi.mlab import pipeline
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from traits.api import HasTraits, on_trait_change, Instance, Property, \
                       Array, Bool, Button, Color, Enum, Float, Str, Tuple
from traitsui.api import View, Item, Group, HGroup

from .transforms import apply_trans


class HeadViewController(HasTraits):
    """
    Set head views for Anterior-Left-Superior coordinate system

    Parameters
    ----------
    system : 'RAS' | 'ALS' | 'ARI'
        Coordinate system described as initials for directions associated with
        the x, y, and z axes. Relevant terms are: Anterior, Right, Left,
        Superior, Inferior.
    """
    system = Enum("RAS", "ALS", "ARI", desc="Coordinate system: directions of "
                  "the x, y, and z axis.")

    right = Button()
    front = Button()
    left = Button()
    top = Button()

    scale = Float(0.16)

    scene = Instance(MlabSceneModel)

    view = View(Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True))

    @on_trait_change('scene.activated')
    def _init_view(self):
        self.scene.parallel_projection = True
        self.sync_trait('scale', self.scene.camera, 'parallel_scale')
        # this alone seems not to be enough to sync the camera scale (see
        # ._on_view_scale_update() method below

    @on_trait_change('scale')
    def _on_view_scale_update(self):
        if self.scene is not None:
            self.scene.camera.parallel_scale = self.scale
            self.scene.render()

    @on_trait_change('top,left,right,front')
    def on_set_view(self, view, _):
        if self.scene is None:
            return

        system = self.system
        kwargs = None

        if system == 'ALS':
            if view == 'front':
                kwargs = dict(azimuth=0, elevation=90, roll= -90)
            elif view == 'left':
                kwargs = dict(azimuth=90, elevation=90, roll=180)
            elif view == 'right':
                kwargs = dict(azimuth= -90, elevation=90, roll=0)
            elif view == 'top':
                kwargs = dict(azimuth=0, elevation=0, roll= -90)
        elif system == 'RAS':
            if view == 'front':
                kwargs = dict(azimuth=90, elevation=90, roll=180)
            elif view == 'left':
                kwargs = dict(azimuth=180, elevation=90, roll=90)
            elif view == 'right':
                kwargs = dict(azimuth=0, elevation=90, roll=270)
            elif view == 'top':
                kwargs = dict(azimuth=90, elevation=0, roll=180)
        elif system == 'ARI':
            if view == 'front':
                kwargs = dict(azimuth=0, elevation=90, roll=90)
            elif view == 'left':
                kwargs = dict(azimuth= -90, elevation=90, roll=180)
            elif view == 'right':
                kwargs = dict(azimuth=90, elevation=90, roll=0)
            elif view == 'top':
                kwargs = dict(azimuth=0, elevation=180, roll=90)
        else:
            raise ValueError("Invalid system: %r" % system)

        if kwargs is None:
            raise ValueError("Invalid view: %r" % view)

        self.scene.mlab.view(distance=None, reset_roll=True,
                             figure=self.scene.mayavi_scene, **kwargs)



class PointObject(HasTraits):
    """Represent 5 marker points"""
    points = Array(float, shape=(None, 3))
    trans = Array(float, shape=(None, None))
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

    @on_trait_change('points,trans')
    def _update_points(self):
        """Update the location of the plotted points"""
        if not hasattr(self, 'src'):
            return

        trans = self.trans
        if np.any(trans):
            if trans.shape == (3, 3):
                pts = np.dot(self.points, trans.T)
            elif trans.shape == (4, 4):
                pts = apply_trans(trans, self.points)
            else:
                raise ValueError("trans must be 3 by 3 or 4 by 4 array")
        else:
            pts = self.points

        self.src.data.points = pts

    @on_trait_change('scene.activated')
    def _plot_points(self):
        """Add the points to the mayavi pipeline"""
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

#        self.sync_trait('points', self.src.data, 'points', mutual=False)
        self.sync_trait('point_scale', self.glyph.glyph.glyph, 'scale_factor')
        self.sync_trait('rgbcolor', self.glyph.actor.property, 'color', mutual=False)
        self.sync_trait('visible', self.glyph, 'visible')

        self.scene.camera.parallel_scale = _scale
