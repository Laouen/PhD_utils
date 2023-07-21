from typing import Union, List
import mne
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from mne.channels import DigMontage, find_ch_adjacency
import warnings


def _get_closest_channels(org_montage: DigMontage, dest_montage: DigMontage):
    ''' Brief
    For each channel in the dest montage, find the closest channel in
    the original montage.

    Parameters
    ----------
    org_montage: DigMontage
        The original montage
    dest_montage: DigMontage
        The destination montage

    Returns
    -------
    closest_channels: List[str]
        The closest channels in the original montage for each channel in the
        destination montage
    dest_channels: List[str]
        The channels in the destination montage in the same order as the closest
        channels in the original montage. i.e. closest_channels[i] is the closest
        channel in the original montage for dest_channels[i]
    
    '''

    # Get original channels potitions
    org_ch_pos = org_montage.get_positions()['ch_pos']
    # |org_ch|
    org_ch_names = np.array(list(org_ch_pos.keys()))
    # |org_ch| x 3
    org_ch_pos = np.array(list(org_ch_pos.values()))

    # Get destination channels positions
    dest_ch_pos = dest_montage.get_positions()['ch_pos']
    # |dest_ch|
    dest_ch_names = np.array(list(dest_ch_pos.keys()))
    # |dest_ch| x 3
    dest_ch_pos = np.array(list(dest_ch_pos.values()))

    # Calculate the matrix of pairwise distances of the channels positions
    # between both montages
    dists = euclidean_distances(dest_ch_pos, org_ch_pos)

    # Pick the closest channel for each channel in the destination montage
    min_idxs = np.argmin(dists, axis=1)
    return org_ch_names[min_idxs], dest_ch_names


def channels_subsampling(inst: Union[mne.Epochs, mne.io.Raw, mne.Evoked],
                         dest_montage:str='standard_1020'):

    # Get the original montage channels names and positions
    org_montage = inst.get_montage()

    # Get the destination montage channels names and positions
    dest_montage_name = dest_montage
    dest_montage = mne.channels.make_standard_montage(dest_montage_name)

    picks, _ = _get_closest_channels(org_montage, dest_montage)

    if len(picks) != len(np.unique(picks)):
        raise ValueError(
            'The closest channels are not unique. There is multiple '
            'channels in the destination montage that have the same '
            'closes channel in the original montage.'
        )

    inst = inst.pick(picks)
    return inst


def channels_recombine_neighborhood(inst: Union[mne.Epochs, mne.io.Raw, mne.Evoked],
                       dest_montage:str='standard_1020', return_groups:bool=False):

    # Get the original montage channels names and positions
    org_montage = inst.get_montage()

    # Get the destination montage channels names and positions
    dest_montage_name = dest_montage
    dest_montage = mne.channels.make_standard_montage(dest_montage_name)

    # Pick for each channel in the destination montage the closest channel and
    # its neightbours from the orginal montage and average them
    org_channels, dest_channels = _get_closest_channels(org_montage, dest_montage)
    ch_adjacency, ch_names = find_ch_adjacency(inst.info, ch_type='eeg')
    org_channels_neighbors = [
        [
            ch_names[ch_idx] 
            for ch_idx in ch_adjacency[ch_names.index(ch)].nonzero()[1]
        ] for ch in org_channels
    ]
    groups = {
        dest_ch: [inst.ch_names.index(ch) for ch in org_ch_neighbors]
        for dest_ch, org_ch_neighbors in zip(dest_channels, org_channels_neighbors)
    }


    # Recombine the channels to map the best possible the destintation montage
    res = mne.channels.combine_channels(
        inst, groups,
        method='mean',
        keep_stim=False,
        drop_bad=False
    )
    res.set_montage(dest_montage)

    if return_groups:
        return res, groups
    return res


def channels_recombine_parcelation(inst: Union[mne.Epochs, mne.io.Raw, mne.Evoked],
                       dest_montage:str='standard_1020', return_groups:bool=False):

    # Get the original montage channels names and positions
    org_montage = inst.get_montage()

    # Get the destination montage channels names and positions
    dest_montage_name = dest_montage
    dest_montage = mne.channels.make_standard_montage(dest_montage_name)

    # For each channel in the original montage, find the closest channel in the
    # destination montage.
    dest_channels, org_channels = _get_closest_channels(dest_montage, org_montage)
    
    groups = {
        dest_ch: [
            inst.ch_names.index(ch)
            for ch in org_channels[dest_channels == dest_ch]
        ] for dest_ch in dest_montage.ch_names
    }

    # All electrode in the destination montage should have at least one
    # electrode from the original montage
    empty_groups = {k:v for k,v in groups.items() if len(v) == 0}
    groups = {k:v for k,v in groups.items() if len(v) > 0}

    if len(empty_groups) > 0:
        warnings.warn(
            f'The following channels in the destination montage do not have '
            f'any corresponding channel in the original montage: '
            f'{list(empty_groups.keys())}. They will be ignored.'
        )

    # Recombine the channels to map the best possible the destintation montage
    res = mne.channels.combine_channels(
        inst, groups,
        method='mean',
        keep_stim=False,
        drop_bad=False
    )
    res.set_montage(dest_montage)

    if return_groups:
        return res, groups, empty_groups
    return res
