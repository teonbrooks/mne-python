"""GUI for coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from pyface.api import error, confirm, YES, NO, CANCEL, ProgressDialog
import traits.api as traits
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import MriHeadFitter



class MriHeadCoreg(traits.HasTraits):
    """
    Mayavi viewer for fitting an MRI to a digitized head shape.

    """
    # views
    right = traits.Button()
    front = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # parameters
    nasion = traits.Array(float, (1, 3))
    rotation = traits.Array(float, (1, 3))
    scale = traits.Array(float, (1, 3), [[1, 1, 1]])
    shrink = traits.Float(1)
    restore_fit = traits.Button()

    # fitting
    fit_scale = traits.Button()
    fit_no_scale = traits.Button()

    # saving
    s_to = traits.String()
    save = traits.Button()

    scene = traits.Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                HGroup('72', 'top', show_labels=False),
                HGroup('right', 'front', 'left', show_labels=False),
                HGroup('fit_scale', 'fit_no_scale', 'restore_fit',
                       show_labels=False),
                HGroup('nasion'),
                HGroup('scale', 'shrink'),
                HGroup('rotation'),
                HGroup('s_to', HGroup('save', show_labels=False)),
                )

    def __init__(self, raw, s_from=None, s_to=None, subjects_dir=None):
        """
        Parameters
        ----------
        raw : str(path)
            path to a raw file containing the digitizer data.
        s_from : str
            name of the source subject (e.g., 'fsaverage').
            Can be None if the raw file-name starts with "{subject}_".
        s_to : str | None
            Name of the the subject for which the MRI is destined (used to
            save MRI and in the trans file's file name).
            Can be None if the raw file-name starts with "{subject}_".
        subjects_dir : None | path
            Override the SUBJECTS_DIR environment variable
            (sys.environ['SUBJECTS_DIR'])

        """
        self.fitter = MriHeadFitter(raw, s_from, s_to, subjects_dir)

        traits.HasTraits.__init__(self)
        self.configure_traits()

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        fig = self.scene.mayavi_scene
        self.scene.disable_render = True
        self.fitter.plot(fig=fig)

        self._text = None
        s_to = self.fitter.s_to
        self.s_to = s_to

        self.front = True
        self.scene.disable_render = False
        self._last_fit = None

    @traits.on_trait_change('fit_scale,fit_no_scale')
    def _fit(self, caller, info2):
        if caller == 'fit_scale':
            self.fitter.fit(method='sr')
        elif caller == 'fit_no_scale':
            self.fitter.fit(method='r')
        else:
            error(self, "Unknown caller for _fit(): %r" % caller, "Error")
            return

        rotation = self.fitter.get_rot()
        scale = self.fitter.get_scale()
        self._last_fit = ([scale], [rotation])
        self.on_restore_fit()

    @traits.on_trait_change('restore_fit')
    def on_restore_fit(self):
        if self._last_fit is None:
            error("No fit has been performed", "No Fit")
            return

        self.scale, self.rotation = self._last_fit
        self.shrink = 1

    @traits.on_trait_change('save')
    def on_save(self):
        s_to = self.s_to

        trans_fname = self.fitter.get_trans_fname(s_to)
        if os.path.exists(trans_fname):
            title = "Replace trans file for %s?" % s_to
            msg = ("A trans file already exists at %r. Replace "
                   "it?" % trans_fname)
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        s_to_dir = self.fitter.get_mri_dir(s_to)
        if os.path.exists(s_to_dir):
            title = "Replace %r?" % s_to
            msg = ("The mri subject %r already exists. Replace "
                   "%r?" % (s_to, s_to_dir))
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        try:
            self.fitter.save(s_to, overwrite=True)
        except Exception as e:
            error(None, str(e))

    @traits.on_trait_change('nasion')
    def on_set_nasion(self):
        args = tuple(self.nasion[0])
        self.fitter.set_nasion(*args)

    @traits.on_trait_change('s_to')
    def on_set_s_to(self, s_to):
        s_from = self.fitter.s_from
        fig = self.scene.mayavi_scene
        if s_to == s_from:
            text = "%s" % s_from
            width = .2
        else:
            text = "%s -> %s" % (s_from, s_to)
            width = .5

        if self._text is None:
            self._text = self.scene.mlab.text(0.01, 0.01, text, figure=fig,
                                              width=width)
        else:
            self._text.text = text
            self._text.width = width

    @traits.on_trait_change('scale,rotation,shrink')
    def on_set_trans(self):
        scale = np.array(self.scale[0])
        scale_scale = (1 - self.shrink) * np.array([1, .4, 1])
        scale *= (1 - scale_scale)
        args = tuple(self.rotation[0]) + tuple(scale)
        self.fitter.set(*args)

    @traits.on_trait_change('top,left,right,front')
    def on_set_view(self, view='front', info=None):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = 150
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'right':
            kwargs.update(azimuth=0, roll=270)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)
