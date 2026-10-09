'''
Utilities for xmax fluence reconstruction.

author:: K. Watanabe <keito.watanabe@kit.edu>
'''

import logging
import os
import time

import functools
import numpy as np
import radiotools.helper as hp
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from scipy.spatial import cKDTree
from NuRadioReco.utilities import units, fft
import NuRadioReco.framework.channel
from NuRadioReco.utilities.trace_utilities import get_channel_fluence

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy

conversion_factor_integrated_signal = scipy.constants.c * scipy.constants.epsilon_0 * units.joule / units.s / units.volt ** 2


def fit_to_lower_envelope_parabola(xmax_vals, chi2_vals, range_around_best_xmax=100.0):
    """
    Fit a parabola to the lower envelope of the chi2 vs Xmax values.

    This replicates what is performed in the "standard" LOFAR Xmax analysis, where a parabola is fit to the lower envelope of the chi2 vs Xmax values to find the minimum chi2 and the corresponding Xmax.

    Parameters
    ----------
    xmax_vals : np.ndarray
        The Xmax values to be fit.
    chi2_vals : np.ndarray
        The chi2 values to be fit.
    range_around_best_xmax : float, optional
        The range around the best fit Xmax to be considered for the parabola fit. Points outside this range will be ignored. Default is 40.0 g/cm^2.

    Returns
    -------
    parabola_fit_param : np.ndarray
        The parameters of the fitted parabola, in the form of a 1D array of length 3, where the first element is the coefficient of the quadratic term, the second element is the coefficient of the linear term, and the third element is the constant term.
    xmax_reconstructed : float
        The reconstructed Xmax value, which is the Xmax value corresponding to the minimum of the fitted parabola. If the fit did not converge, this will be None.
    fit_converged : bool
        Whether the fit converged or not. If the fit did not converge, this will be False, and the xmax_reconstructed will be None.
    """

    # select lower envelope points.
    xmax_vals_nozeros = xmax_vals[xmax_vals > 0]
    chi2_vals_nozeros = chi2_vals[xmax_vals > 0]
    points_selection = np.zeros(len(xmax_vals_nozeros), dtype=bool)

    xmax_best_fit_estimate = xmax_vals_nozeros[chi2_vals_nozeros.argmin()]
    for i, xmax in enumerate(xmax_vals_nozeros):
        if np.abs(xmax - xmax_best_fit_estimate) > range_around_best_xmax:
            continue  # do not include points that are too far from the best fit estimate, since they are not relevant for the parabola fit and can bias it.
        all_higher_xmax = np.where(xmax_vals_nozeros > xmax)
        all_lower_xmax = np.where(xmax_vals_nozeros < xmax)

        all_higher_xmax_chi2 = chi2_vals_nozeros[all_higher_xmax]
        all_lower_xmax_chi2 = chi2_vals_nozeros[all_lower_xmax]
        if np.sum(all_higher_xmax_chi2 < chi2_vals_nozeros[i]) == 0:
            points_selection[i] = True
        if np.sum(all_lower_xmax_chi2 < chi2_vals_nozeros[i]) == 0:
            points_selection[i] = (
                True  # so True if points with lower chi^2 are either all to the left or to the right in a plot vs Xmax
            )

    # do parabola fit
    xmax_selected = xmax_vals_nozeros[(points_selection == True)]
    chi2_selected = chi2_vals_nozeros[
        (points_selection == True)
    ]  # put into instance variable for plotting later
    fit_order = 2

    fit_converged = False
    if not np.sum(points_selection) > fit_order:  # ensure n_sample > ndof
        return None, None, fit_converged

    # fit to a second order polynomial
    parabola_fit_param, _, _, _, _ = np.polyfit(
        xmax_selected, chi2_selected, fit_order, full=True
    )

    # evaluate the parabola fit, and find the minimum. This is the reconstructed Xmax.
    parabola_fit = np.poly1d(parabola_fit_param)

    # check minimum, check shape of parabola
    derivative = parabola_fit.deriv()
    zero_point = np.roots(derivative)[0]
    if np.abs(zero_point - xmax_best_fit_estimate) < range_around_best_xmax:
        fit_converged = True
        xmax_reconstructed = zero_point
    else:
        xmax_reconstructed = None
    # also test for sign of parabola`
    if parabola_fit_param[0] < 0:
        fit_converged = False
        xmax_reconstructed = None

    return parabola_fit_param, xmax_reconstructed, fit_converged

def get_lower_envelope_indices(xmax_vals, chi2_vals, range_around_best_xmax = 100.0):
    """
    Get the indices of the points that are part of the lower envelope of the chi2 vs Xmax values.
    """
    # select lower envelope points. 
    xmax_vals_nozeros = xmax_vals[xmax_vals > 0]
    chi2_vals_nozeros = chi2_vals[xmax_vals > 0]
    points_selection = np.zeros(len(xmax_vals_nozeros), dtype=bool)

    xmax_best_fit_estimate = xmax_vals_nozeros[chi2_vals_nozeros.argmin()]
    for i, xmax in enumerate(xmax_vals_nozeros):
        if np.abs(xmax - xmax_best_fit_estimate) > range_around_best_xmax:
            continue # do not include points that are too far from the best fit estimate, since they are not relevant for the parabola fit and can bias it.
        all_higher_xmax = np.where(xmax_vals_nozeros > xmax)
        all_lower_xmax = np.where(xmax_vals_nozeros < xmax)

        all_higher_xmax_chi2 = chi2_vals_nozeros[all_higher_xmax]
        all_lower_xmax_chi2 = chi2_vals_nozeros[all_lower_xmax]
        if np.sum(all_higher_xmax_chi2 < chi2_vals_nozeros[i]) == 0:
            points_selection[i] = True 
        if np.sum(all_lower_xmax_chi2 < chi2_vals_nozeros[i]) == 0:
            points_selection[i] = True # so True if points with lower chi^2 are either all to the left or to the right in a plot vs Xmax

    return points_selection == True

def plot_chi2_parabola(
    xmax_vals,
    red_chi2_vals,
    fit_param_vals,
    parabola_fit_param,
    xmax_reconstructed,
    lower_envelope_indices,
    xmax_fit_window = 100,
    xmax_inset_window = 50,
    mc_xmax = None,
    fit_param_label = "scale_factor",
    parab_color = 'r',
    save_path=None,
    fig = None
):
    """
    Plot the chi2 vs Xmax values, the fitted parabola, and the reconstructed Xmax.

    Parameters
    ----------
    xmax_vals : np.ndarray
        The Xmax values to be plotted.
    red_chi2_vals : np.ndarray
        The reduced chi2 values to be plotted.
    fit_param_vals : np.ndarray
        The parameter to be plotted on the color axis. This is typically the scale factor squared, but can be any parameter that is relevant for the reconstruction.
    parabola_fit_param : np.ndarray
        The parameters of the fitted parabola, in the form of a 1D array of length 3, where the first element is the coefficient of the quadratic term, the second element is the coefficient of the linear term, and the third element is the constant term.
    xmax_reconstructed : float
        The reconstructed Xmax value, which is the Xmax value corresponding to the minimum of the fitted parabola. If the fit did not converge, this will be None.
    lower_envelope_indices : np.ndarray
        The indices of the points that are part of the lower envelope of the chi2 vs Xmax values. This is used to highlight the points that were used for the parabola fit.
    xmax_fit_window : float, optional
        The window around the reconstructed Xmax to be plotted for the parabola fit. Default is 100 g/cm^2.
    xmax_inset_window : float, optional
        The window around the reconstructed Xmax to be plotted for the inset. Default is 50 g/cm^2.
    mc_xmax : float, optional
        The true Xmax value to be plotted. If None, it will not be plotted. Default is None.
    fit_param_label : str, optional
        The label for the fit parameter to be plotted on the color axis. Default is "scale_factor".

        Other accepted options are 'core_x', 'core_y', 'scale_factor_sq'.
    save_path : str, optional
        The path to save the plot. If None, it will not be saved. Default is None.
    fig : matplotlib.figure.Figure, optional
        The figure to be used for the plot. If None, a new figure will be created. Default is None.

        This is useful if you want to plot additional reconstruction results on the same figure, e.g. for different signal chain settings.
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,8))
    else:
        ax = fig.gca()

    ax_inset = ax.inset_axes([0.08, 0.58, 0.35, 0.35])

    # choose different color maps for different fit parameters
    if fit_param_label == "scale_factor":
        cbar_cmap = cm.viridis
        cbar_label = "Scale factor"
    elif fit_param_label == "core_x":
        cbar_cmap = cm.plasma
        cbar_label = "Core x [m]"
    elif fit_param_label == "core_y":
        cbar_cmap = cm.inferno
        cbar_label = "Core y [m]"
    elif fit_param_label == "scale_factor_sq":
        cbar_cmap = cm.viridis
        cbar_label = "Scale factor squared"
    else:
        raise ValueError(f"Unknown fit parameter label: {fit_param_label}. Accepted options are 'scale_factor', 'core_x', 'core_y', 'scale_factor_sq'.")

    cbar_norm = mcolors.Normalize(vmin=np.min(fit_param_vals), vmax=np.max(fit_param_vals))

    for axis in (ax, ax_inset):
        axis.scatter(
            xmax_vals,
            red_chi2_vals,
            facecolors="none",
            edgecolors=cbar_cmap(cbar_norm(fit_param_vals)),
            linewidths=1.2,
            s=50,
            zorder=4,
        )
        axis.scatter(
            xmax_vals[lower_envelope_indices],
            red_chi2_vals[lower_envelope_indices],
            c=fit_param_vals[lower_envelope_indices],
            cmap=cbar_cmap,
            norm = cbar_norm,
            s=50,
            zorder=5,
        )
        if parabola_fit_param is not None and xmax_reconstructed is not None:
            parabola_fit = np.poly1d(parabola_fit_param)
            
            x_fit = np.linspace(xmax_reconstructed - xmax_fit_window, xmax_reconstructed + xmax_fit_window, 100)
            min_chi2 = parabola_fit(xmax_reconstructed)
            axis.plot(
                x_fit,
                parabola_fit(x_fit),
                color=parab_color,
                linestyle="-",
                linewidth=2,
                label="Parabola fit",
                zorder=6,
            )
            axis.axvline(
                xmax_reconstructed,
                color=parab_color,
                linestyle=":",
                linewidth=2,
                label=f"Reconstructed Xmax = {xmax_reconstructed:.1f} g/cm²" + '\n' + f"Min chi² = {min_chi2:.2f}",
                zorder=7,
            )
            
        if mc_xmax is not None:
            axis.axvline(
                mc_xmax,
                color="k",
                linestyle="--",
                linewidth=2,
                label=f"MC Xmax = {mc_xmax:.1f} g/cm²",
                zorder=8,
            )
            
        ax.set_xlabel(r"$X_\mathrm{max}$ / g cm$^{-2}$", fontsize=14)
        ax.set_ylabel(r"$\chi^2 / \mathrm{dof}$", fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.set_ylim(ymin=0)
        
        # inset plot
        xmax_center = np.mean(xmax_vals[lower_envelope_indices]) if xmax_reconstructed is None else xmax_reconstructed
        ax_inset.set_xlim(xmax_center - xmax_inset_window, xmax_center + xmax_inset_window)
        y_window = xmax_inset_window / 2.5
        ylo_bound, yhi_bound = xmax_center - y_window, xmax_center + y_window
        ax_inset.set_ylim(ylo_bound, yhi_bound)
        
        ax_inset.tick_params(labelsize=8)

        sm = cm.ScalarMappable(norm=cbar_norm, cmap=cbar_cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.04)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        
        if save_path is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
        
        return fig

def plot_reco_histograms(
    reco_red_chi2_vals_successful,
    reco_core_x_successful,
    reco_core_y_successful,
    reco_scale_factor_successful,
    mc_core_x=None,
    mc_core_y=None,
    save_path=None
):
    """
    Plot histograms of the reconstructed parameters.

    Parameters
    ----------
    reco_red_chi2_vals_successful : np.ndarray
        The reduced chi2 values of the successful reconstructions.
    reco_core_x_successful : np.ndarray
        The reconstructed core x positions of the successful reconstructions.
    reco_core_y_successful : np.ndarray
        The reconstructed core y positions of the successful reconstructions.
    reco_scale_factor_successful : np.ndarray
        The reconstructed scale factors of the successful reconstructions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].hist(reco_red_chi2_vals_successful, bins=15, color='b', alpha=0.7)
    axes[0].set_title('Reduced Chi² Histogram')
    axes[0].set_xlabel('Reduced Chi²')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')  # Set y-axis to log scale for better visibility of low counts
    axes[0].set_xlim(left=0)  # Set x-axis lower limit to 0
    axes[0].axvline(x=1, color='k', linestyle='--', label='Chi² = 1')  # Add a vertical line at Chi² = 1
    axes[0].legend()

    axes[1].hist(reco_core_x_successful, bins=15, color='g', alpha=0.7)
    axes[1].set_title('Reconstructed Core X Histogram')
    axes[1].set_xlabel('Core X (m)')
    axes[1].set_ylabel('Count')
    if mc_core_x is not None:
        axes[1].axvline(x=mc_core_x, color='k', linestyle='--', label=f'MC Core X = {mc_core_x:.2f} m')  # Add a vertical line at MC Core X
        axes[1].legend()

    axes[2].hist(reco_core_y_successful, bins=15, color='r', alpha=0.7)
    axes[2].set_title('Reconstructed Core Y Histogram')
    axes[2].set_xlabel('Core Y (m)')
    axes[2].set_ylabel('Count')
    if mc_core_y is not None:
        axes[2].axvline(x=mc_core_y, color='k', linestyle='--', label=f'MC Core Y = {mc_core_y:.2f} m')  # Add a vertical line at MC Core Y
        axes[2].legend()

    axes[3].hist(reco_scale_factor_successful, bins=15, color='c', alpha=0.7)
    axes[3].set_title('Reconstructed Scale Factor Histogram')
    axes[3].set_xlabel('Scale Factor')
    axes[3].set_ylabel('Count')
    axes[3].set_yscale('log')  # Set y-axis to log scale for better visibility of low counts
    axes[3].axvline(x=1, color='k', linestyle='--', label='Scale Factor = 1')  # Add a vertical line at Scale Factor = 1
    axes[3].legend()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        
    return fig
    
    
class miniChannelFluenceCalculator:
    """
    A class to calculate the fluence (on channel level) in a fast fashion.
    
    This is needed since the method in the standard approach is too slow for any meaningful reconstruction.
    
    This class takes in an event, generated with readCoREAS/SMIETDetector, and :
    - Applies efield to voltage conversion
    - Calculates the fluence for each channel
    
    And the end result stores the fluence as an array of shape (n_stations, n_channel_groups).
    """
    def __init__(self) -> None:
        self.detector = None
        
        # antenna conversion-specific parameters
        self.antenna_pattern = None
        self.antenna_orientation = None
        
        # efield specific parameters
        self.__freqs = None
        
        # fluence-specific parameters
        self.data_antenna_positions = None
        self.signal_window_length = None
    
    def begin(self, detector, data_ant_positions, selected_station_channel_ids = None, signal_window_length = 55 * units.ns):
        """
        Initialize the calculator with the detector information.
        
        Also sets up the antenna model, and pattern so that E -> V can be done much faster than in the standard approach, where the antenna pattern is loaded for each channel and each event.
        
        This begin function should only contain properties that are independent of each simulation, and can be reused for all events. This is important for speed, since this function is called only once per simulation, while the run function is called for each event.
        
        Parameters
        ----------
        detector : NuRadioReco.detector.Detector
            The detector object containing the antenna information.
        data_ant_positions : np.ndarray
            The positions of the data antennas.
        data_event : NuRadioReco.framework.event.Event
            The event object containing the data to be processed.
        """
        self.detector = detector
        self.data_antenna_positions = data_ant_positions
        self.signal_window_length = signal_window_length

        # here we assume that the antenna model is the same for all channels and stations. For this particular purpose, and for most air-shower experiments, this is indeed the case. If this is not the case, then this class needs to be modified to handle different antenna models for different channels. (TODO)
        
        # for now, we store a single antenna model, with a single station and channel
        station_id_to_use = self.detector.get_station_ids()[0]
        channel_ids_to_use = self.detector.get_channel_ids(station_id_to_use)
        
        antenna_model = self.detector.get_antenna_model(station_id_to_use, channel_ids_to_use[0])
        
        # now import the antenna pattern once
        antenna_pattern_provider = antennapattern.AntennaPatternProvider()
        self.antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        
        # here we get an index of the antenna positions, indexed by group id, station ID, and channel ID, as well as the antenna orientation that it should be used.
        # allows for fast lookup of the antenna position and orientation for each channel, without having to query the detector for each channel in each event.
        self._groups = []   # one entry per antenna = per channel group
        
        for station_id, ch_ids in selected_station_channel_ids.items():
            by_group = {}
            for ch in ch_ids:
                by_group.setdefault(self.detector.get_channel_group_id(station_id, ch), []).append(ch)
            for grp_id, chs in by_group.items():
                pos = self.detector.get_absolute_position(station_id) + self.detector.get_relative_position(station_id, chs[0])
                orients = [self.detector.get_antenna_orientation(station_id, ch) for ch in chs]
                self._groups.append(dict(station_id=station_id, group_id=grp_id,
                                        channel_ids=chs, position=pos, orientations=orients))
        # this is a dictionary that maps (station_id, group_id) to the index of the group in self._groups. This allows for fast lookup of the group index for each channel in each event.
        self._group_index = {(g["station_id"], g["group_id"]): i
                             for i, g in enumerate(self._groups)}

        # here is the antenna positions, indexed by the group index. 
        # we use a KDTree to find the nearest neighbor in the simulated antenna positions for each data antenna position, and then sort the fluences and antenna positions accordingly. This is important for the reconstruction, since we want to compare the simulated fluences with the data fluences.
        self._positions = np.array([g["position"] for g in self._groups])
        _, self._kdtree_indices = cKDTree(self._positions).query(data_ant_positions)   # once, not per call

        self._row_by_first_ch = {(g["station_id"], g["channel_ids"][0]): i
                         for i, g in enumerate(self._groups)}

        
        # in principle only need two channels, but for completeness, we store the orientation for all channels. This is needed to apply the antenna response to the efield.
        # its okay to do this here since its a preprocessing step
        self.antenna_orientation = [
            self.detector.get_antenna_orientation(station_id_to_use, channel_ids_to_use[0]),
            self.detector.get_antenna_orientation(station_id_to_use, channel_ids_to_use[1])
        ]
        
    def get_fluence(self, event):
        """
        Calculate the fluence for each channel in the event.
        
        Does the following:
        - For each station, get the simulated efield and the arrival direction (zenith, azimuth)
        - For each channel, get the antenna orientation and apply the antenna response to the efield
        - Calculate the fluence for each channel, and store it in a list
        
        Parameters
        ----------
        event : NuRadioReco.framework.event.Event
            The event object containing the simulated efields.
        selected_station_channel_ids : dict, optional
            A dictionary containing the channel IDs to be used for each station. If None, all channels will be used. The keys are the station IDs, and the values are lists of channel IDs.
            
        Returns
        -------
        sim_fluences_sorted : np.ndarray
            The fluences for each channel, sorted by the antenna positions. Shape is (n_antennas).
        sim_ant_pos_sorted : np.ndarray
            The antenna positions for each channel, sorted by the antenna positions. Shape is (n_antennas, 3).
        """
        time_start = time.time()
        # get out the zenith and azimuth angles from a single electric field object, since we assume that all channels have the same arrival direction. This is indeed the case for air showers, but not necessarily for other types of events.
        station_id_to_use = self.detector.get_station_ids()[0]
        sim_station = event.get_station(station_id_to_use).get_sim_station()
        channel_id_to_use = self.detector.get_channel_ids(sim_station.get_id())[0]
        
        efield_to_use = sim_station.get_electric_fields()[0]
        zenith = efield_to_use[efp.zenith]
        azimuth = efield_to_use[efp.azimuth]
        sampling_rate = efield_to_use.get_sampling_rate()
        
        det_sampling_rate = self.detector.get_sampling_frequency(sim_station.get_id(), channel_id_to_use)
        n_time_bins = efield_to_use.get_number_of_samples()
        
        n_new = int(round(n_time_bins * det_sampling_rate / sampling_rate))
        n_f = n_new // 2 + 1
        self.__freqs = efield_to_use.get_frequencies()[:n_f]
        
        # get the antenna response at at given zenith & azimuth angle. This is done in a vectorized way, so that we can get the response for all channels at once. Only valid since we assume that all channels in a group have the same orientation. This is much faster than doing it for each channel separately.
        geo_param_key = (zenith, azimuth, n_time_bins, sampling_rate)
        if geo_param_key != getattr(self, "_R_key", None):
            vX = self.antenna_pattern.get_antenna_response_vectorized(self.__freqs, zenith, azimuth, *self.antenna_orientation[0])
            vY = self.antenna_pattern.get_antenna_response_vectorized(self.__freqs, zenith, azimuth, *self.antenna_orientation[1])
            self._R = np.array([[vX['theta'], vX['phi']], [vY['theta'], vY['phi']]])
            self._R_key = geo_param_key
        R = self._R
        
        fluence = np.zeros(len(self._groups))
        t = np.arange(n_new) / det_sampling_rate
        norm = conversion_factor_integrated_signal / det_sampling_rate

        for station in event.get_stations():
            sid = station.get_id()
            efields = station.get_sim_station().get_electric_fields()
            rows = np.array([self._row_by_first_ch[(sid, ef.get_channel_ids()[0])] for ef in efields])

            # only eTheta, ePhi are needed: drops 1/3 of the memory
            E = np.stack([ef.get_trace()[1:3] for ef in efields])         # (n_g, 2, n_t)
            Ef = fft.time2freq(E, sampling_rate)[..., :n_f]                # (n_g, 2, n_f)
            del E

            V = R[None, :, 0] * Ef[:, None, 0] + R[None, :, 1] * Ef[:, None, 1]   # (n_g, 2 ch, n_f)
            del Ef
            v = fft.freq2time(V, det_sampling_rate, n=n_new)               # (n_g, 2 ch, n_new)

            tpk = t[np.argmax(np.abs(v), axis=-1)][..., None]
            mask = np.abs(t - tpk) < 0.5 * self.signal_window_length
            fluence[rows] = (v**2 * mask).sum(axis=(-1, -2)) * norm

        # return fluence[self._kdtree_indices], self._positions[self._kdtree_indices]

        
        # time_start = time.time()
        # E = np.zeros((len(self._groups), 3, n_time_bins))  # shape (n_groups, 3, n_time_bins)
        # for station in event.get_stations():
        #     sim_station = station.get_sim_station()
        #     for ef in sim_station.get_electric_fields():          # one per group
        #         grp_id = self.detector.get_channel_group_id(station.get_id(), ef.get_channel_ids()[0])
        #         E[self._group_index[(station.get_id(), grp_id)]] = ef.get_trace()
        # time_end = time.time()
        # print(f"Time to get out the efields: {time_end - time_start:.4f} s")
        
        # time_start = time.time()
        # Ef = fft.time2freq(E, sampling_rate)[..., :n_f]                        # (n_ant, 3, n_f)
        # V = R[None, :, 0] * Ef[:, None, 1] + R[None, :, 1] * Ef[:, None, 2]   # (n_ant, 2, n_f)

        # v = fft.freq2time(V, det_sampling_rate, n=n_new)                                  # (n_ant, 2, n_new)
        # time_end = time.time()
        # print(f"Time to get out the voltages: {time_end - time_start:.4f} s")
        
        # time_start = time.time()
        # t = np.arange(n_new) / det_sampling_rate
        # tpk = t[np.argmax(np.abs(v), axis=-1)][..., None]
        # mask = np.abs(t - tpk) < 0.5 * self.signal_window_length
        # fluence = (v**2 * mask).sum(-1).sum(-1) / det_sampling_rate * conversion_factor_integrated_signal   # (n_ant,)
        # time_end = time.time()
        # print(f"Time to get out the fluences: {time_end - time_start:.4f} s")

        # sim_fluences = []
        # sim_ant_pos = []
        # for station in event.get_stations():
        #     sim_station = station.get_sim_station()
        #     sim_station_id = sim_station.get_id()
            
        #     station_pos = self.detector.get_absolute_position(station.get_id())
            
        #     channel_ids_to_use = self.detector.get_channel_ids(sim_station_id) if selected_station_channel_ids == None else selected_station_channel_ids[station.get_id()]
            
        #     sim_efields = [ef for ef in sim_station.get_electric_fields() if ef.has_channel_ids(channel_ids_to_use)]
            
            
            
            # channel_group_id = None
            # fluence_per_channel_group = 0.0
            
            # for sim_efield in sim_efields:
                
            #     efield_fft = sim_efield.get_frequency_spectrum()
                
            #     print(sim_efield.get_channel_ids())
                
            #     for channel_id in sim_efield.get_channel_ids():
                    
            #         voltage_fft = vel[0] * efield_fft[1] + vel[1] * efield_fft[2]
            #         voltage_fft[np.where(self.__freqs < 5 * units.MHz)] = 0.

            #     #     voltage_ffts.append(voltage_fft)
                    
            #     channel_spectrum = np.sum(np.array(voltage_ffts), axis=0)
                
            #     # store the fluence in this channel object, since we want to get 
            #     # out the trace & apply resampling to it
            #     channel_obj = NuRadioReco.framework.channel.Channel(channel_id, channel_group_id=channel_group_id)
            #     channel_obj.set_frequency_spectrum(channel_spectrum, sampling_rate)
                
            #     # now run the resampler to resample to detector sampling rate
            #     channel_obj.resample(sampling_rate=det_sampling_rate)
                
            #     # calculate fluence
            #     traces = channel_obj.get_trace()
            #     times = channel_obj.get_times()
            #     peak_time = times[np.argmax(np.abs(traces))]
            #     time_window = (times > peak_time - 0.5 * self.signal_window_length) & (times < peak_time + 0.5 * self.signal_window_length)
            #     fluence = np.sum(traces[time_window]**2) * (times[1] - times[0]) * conversion_factor_integrated_signal
                
            #     fluence_per_channel_group += fluence
        
        
        # # convert to numpy arrays
        # sim_fluences = np.array(sim_fluences)
        # sim_ant_pos = np.array(sim_ant_pos)
        # print(f"sim_fluences: {np.max(sim_fluences)}")
        
        # # finally sort the fluences and antenna positions by the antenna positions, so that they are in the same order as the data antenna positions. This is important for the reconstruction, since we want to compare the simulated fluences with the data fluences.
        # # we use a KDTree to find the nearest neighbor in the simulated antenna positions for each data antenna position, and then sort the fluences and antenna positions accordingly.
        # tree = cKDTree(sim_ant_pos)
        # _, indices = tree.query(self.data_antenna_positions)
        # sim_fluences_sorted = sim_fluences[indices]
        # sim_ant_pos_sorted = sim_ant_pos[indices]
        # print(f"sim_fluences sorted: {np.max(sim_fluences_sorted)}")
        
        # return sim_fluences_sorted, sim_ant_pos_sorted
        return fluence[self._kdtree_indices], self._positions[self._kdtree_indices]
    
    def end(self):
        """
        Finalize the calculator.
        """
        pass