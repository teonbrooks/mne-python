import os.path as op

import mne
from mne.forward import make_field_map, transform_instances


path = mne.datasets.sample.data_path()
evoked_fname = op.join(path, 'MEG', 'sample', 'sample_audvis-ave.fif')
trans_fname = op.join(path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
subjects_dir = op.join(path, 'subjects')

condition = 'Left Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(-0.2, 0.0))

# let's assume you have multiple runs and your subject moved :(
evoked_rot = evoked.copy()
trans_rot = evoked_rot.info['dev_head_t']['trans']
trans_rot[:] = mne.transforms.translation(x=0, y=0, z=0.05).dot(trans_rot)

# now let's put it back in place
evoked_old = evoked_rot.copy()
transform_instances([evoked, evoked_rot])

# let's make a before and after map
maps = make_field_map(evoked_old, trans_fname, subject='sample',
                      subjects_dir=subjects_dir, meg_surf='helmet')
evoked_old.plot_field(maps, time=0.11)

evoked_rot = evoked.copy()
trans_rot = evoked_rot.info['dev_head_t']['trans']
trans_rot[:] = mne.transforms.translation(x=0, y=0, z=0.05).dot(trans_rot)

maps = make_field_map(evoked_rot, trans_fname, subject='sample',
                      subjects_dir=subjects_dir,  meg_surf='helmet')
evoked_rot.plot_field(maps, time=0.11)
