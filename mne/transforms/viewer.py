"""Mayavi/traits GUI elements"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from mayavi.tools.mlab_scene_model import MlabSceneModel
from traits.api import HasTraits, on_trait_change, Instance, Button, Enum, \
                       Float
from traitsui.api import View, Item, Group, HGroup



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
