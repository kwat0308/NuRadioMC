"""
CoREAS Fluence Reconstruction module for NuRadioReco.

.. moduleauthor:: Keito Watanabe <keito.watanabe@kit.edu>
"""

import logging
import os

import numpy as np
import radiotools.helper as hp

import NuRadioReco.modules.efieldToVoltageConverterMini
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.io.coreas.readCoREASDetector
from NuRadioReco.modules.io.coreas import coreas, coreasInterpolator
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

from NuRadioReco.modules.base.module import register_run

from NuRadioReco.utilities import units
from NuRadioReco.utilities.LOFAR.fluence_reco_utilities import (
    fit_to_lower_envelope_parabola,
    get_lower_envelope_indices,
    plot_chi2_parabola,
    plot_reco_histograms,
    miniChannelFluenceCalculator
)
import time

import scipy.optimize as opt


class CoREASFluenceReconstructor:
    """
    Reconstruction module of Xmax using CoREAS fluence maps.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.LOFAR.CoREASFluenceReconstructor")
        self.input_files = None
        self.sim_xmaxs = None
        self.__settings = {}

    def begin(
        self,
        input_files=None,
        site="lofar",
        fluence_window_ns=55,
        coicidence_window_ns=20,
        pass_band_MHz=(30, 80),
        pre_post_pulse_time_ns=1000,
        bandpass_half_hann_percent=0.1,
        bandpass_gaussian_roll_width_MHz=2.5,
        xmax_parabola_fit_range_gcm2=100.0,
        output_directory=None,
        hybrid_shower_name="LORA",
        logger_level=logging.NOTSET,
    ):
        """
        Initialize the module.
        """
        self.input_files = input_files

        # add all arguments into settings
        self.__settings = {
            "site": site,
            "fluence_window": fluence_window_ns * units.ns,
            "coincidence_window": coicidence_window_ns * units.ns,
            "pass_band": [pass_band_MHz[0] * units.MHz, pass_band_MHz[1] * units.MHz],
            "pre_pulse_time": pre_post_pulse_time_ns * units.ns,
            "post_pulse_time": pre_post_pulse_time_ns * units.ns,
            "bandpass_half_hann_percent": bandpass_half_hann_percent,
            "bandpass_gaussian_roll_width": bandpass_gaussian_roll_width_MHz
            * units.MHz,
            "xmax_parabola_fit_range": xmax_parabola_fit_range_gcm2
            * units.g
            / units.cm**2,
            "output_directory": output_directory,
            "parabola_color" : 'r', # fixed for Coreas ,
            "hybrid_shower_name": hybrid_shower_name,
        }
        
        detector_logger = logging.getLogger("NuRadioReco.detector")
        detector_logger.setLevel(logging.CRITICAL)  # suppress detector logging, since we don't need it here

        # Initialize the modules
        self.efieldToVoltageConverter = (
            NuRadioReco.modules.efieldToVoltageConverterMini.efieldToVoltageConverterMini(
                log_level=logging.CRITICAL
            )
        )
        self.efieldToVoltageConverter.begin(
            debug=False,
            pre_pulse_time=self.__settings["pre_pulse_time"],
            post_pulse_time=self.__settings["post_pulse_time"],
            skip_trace_padding=False,  # skip this for the fluence calculation, its not strictly necessary
        )

        self.channelBandPassFilter = (
            NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
        )
        self.channelBandPassFilter.begin()
        self.channelBandPassFilter.logger.setLevel(logging.CRITICAL)

        self.channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
        self.channelResampler.begin()

        self.channelSignalReconstructor = (
            NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor(
                log_level=logging.CRITICAL
            )
        )
        self.channelSignalReconstructor.begin(
            signal_window_length=self.__settings["fluence_window"],
            coincidence_window_size=self.__settings[
                "coincidence_window"
            ],  # not used in this module, but required for the begin() function
        )

        self.coreas_reader = (
            NuRadioReco.modules.io.coreas.readCoREASDetector.readCoREASDetector()
        )
        self.coreas_reader.logger.setLevel(logging.CRITICAL)
        
        self.miniChannelFluenceCalculator = miniChannelFluenceCalculator()

        self.logger.setLevel(logger_level)

    def _read_event(self, coreas_hdf5_file: str):
        """
        Read the event from the input files.

        Again replaces the readCoREASDetector.begin() function, which is not suitable for reading CoREAS HDF5 files generated with the LOFAR library due to the vertical core coordinate. Instead, we read the event directly from the HDF5 file and set the core coordinate to be vertical.

        Parameters
        ----------
        coreas_hdf5_file : str
            The path to the CoREAS HDF5 file.

        Returns
        -------
        NuRadioReco.framework.event.Event
            The event read from the CoREAS HDF5 file.
        """
        corsika_event = coreas.read_CORSIKA7(
            coreas_hdf5_file, site=self.__settings["site"]
        )

        # here we force vertical core coordinate
        sim_core = corsika_event.get_first_sim_shower().get_parameter(shp.core)
        corsika_event.get_first_sim_shower().set_parameter(
            shp.core,
            np.array(
                [
                    sim_core[0],
                    sim_core[1],
                    corsika_event.get_first_sim_shower().get_parameter(
                        shp.observation_level
                    ),
                ]
            ),
        )

        interpolator = coreasInterpolator.coreasInterpolator(corsika_event, log_level=logging.CRITICAL)
        interpolator.initialize_efield_interpolator(
            interp_lowfreq=self.__settings["pass_band"][0],
            interp_highfreq=self.__settings["pass_band"][1],
        )
        self.coreas_reader.coreas_interpolator = interpolator  # skip begin() function because HDF5 does not have good CoreCoordinateVertical

        self.coreas_reader._readCoREASDetector__corsika_evt = corsika_event

        return corsika_event

    def _perform_fluence_fit(
        self,
        sim_event,
        det,
        data_fluence,
        data_fluence_err,
        core_guess,
        selected_station_channel_ids=None,
    ):
        """
        Perform the fluence fit to the data.

        This does the following:
        - generates the event for a given "fitted" core position (using readCoREASDetector)
        - applies the signal chain (Efield -> Voltage -> Bandpass -> Resample -> Signal Properties)
        - calculates the fluence per station
        - calculates the chi2 between the data fluence and the simulated fluence

        Parameters
        ----------
        sim_event : NuRadioReco.framework.event.Event
            The event to be processed. This is the *sim* event, not the *data* event.
        det : NuRadioReco.detector.Detector
            The detector to be used for the reconstruction.
        data_fluence : dict
            The fluence of the event per station, with the key corresponding to the channel group ID.

        Returns
        -------
        dict
            The fit parameters of the fluence fit, with the key corresponding to the channel group ID.
        """

        def objective_function(x):
            """
            Objective function for the fluence fit.

            This returns the chi2 value:

            chi2 = sum((data_fluence[i] - scale_factor * sim_fluence[i])**2 / data_fluence_err[i]**2 for i in range(len(data_fluence)))

            Unforunately need to be nested function since we use the properties of the sim_event and det objects, which are not available in the global scope.

            Parameters
            ----------
            x : list
                The fit parameters, with the first two elements corresponding to the core position (x, y) and the third element corresponding to the scale factor squared.
            """
            core_x, core_y, scale_factor_sq = x
            core = np.array(
                [
                    core_x,
                    core_y
                ]
            )

            time_start = time.time()
            for sim_evt in self.coreas_reader.run(
                det, [core], selected_station_channel_ids=selected_station_channel_ids
            ):
                time_end = time.time()
                # print(f"Time to generate event for core position ({core_x}, {core_y}): {time_end - time_start:.4f} s")
                sim_fluence, _ = self.miniChannelFluenceCalculator.get_fluence(sim_evt)
            
            # calculate the chi2 between the data fluence and the simulated fluence
            chi2 = np.sum((data_fluence - scale_factor_sq * sim_fluence) ** 2 / data_fluence_err**2)
            
            return chi2

        try:
            res = opt.minimize(
                lambda x: objective_function(x), x0=[core_guess[0], core_guess[1], 1.0], method="CG", bounds=[(-600 * units.m, 600 * units.m), (-600 * units.m, 600 * units.m), (0, None)]

            )
        except Exception as e:
            self.logger.warning(
                f"Fluence fit failed for event {sim_event.get_id()}. Error: {e}", exc_info=True
            )
            return {
                "core_x": np.nan,
                "core_y": np.nan,
                "scale_factor_sq": np.nan,
                "chi2": np.nan,
                "n_dof": np.nan,
                "success": False,
            }
            
        ndof = len(data_fluence) - 3  # number of data points minus number of fit parameters
            
        print("fit results:", res.x)
        print("fit success:", res.success)
        print("fit red chi2:", res.fun / ndof)
        print("fit nfev:", res.nfev)
        
        
        self.logger.info(
            f"Fluence fit succeeded for event {sim_event.get_id()}. Fit parameters: core_x={res.x[0]:.2f}, core_y={res.x[1]:.2f}, scale_factor_sq={res.x[2]:.2f}, chi2={res.fun:.2f}, n_dof={ndof}, success={res.success}"
        )

        # store the fit parameters in a dictionary
        fit_results = {
            "core_x": res.x[0],
            "core_y": res.x[1],
            "scale_factor_sq": res.x[2],
            "chi2": res.fun,
            "n_dof": ndof,  # number of data points minus number of fit parameters
            "success": res.success,
        }

        return fit_results

    def _get_data_fluences(self, event, det, selected_station_channel_ids=None):
        """
        Get the fluence of the data event per station.

        Parameters
        ----------
        event : NuRadioReco.framework.event.Event
            The event to be processed. This is the *data* event, not the *sim* event.
        det : NuRadioReco.detector.Detector
            The detector to be used for the reconstruction.
        selected_station_channel_ids : list of int, optional
            The list of channel IDs to be processed. If None, all channels will be processed.

        Returns
        -------
        data_fluences : numpy.ndarray
            The fluence of the data event per station.
        data_fluences_err : numpy.ndarray
            The error on the fluence of the data event per station.
        data_ant_pos : numpy.ndarray
            The positions of the antennas in the data event.
        """
        data_fluences = []
        data_fluences_err = []
        data_ant_pos = []
        for station in event.get_stations():

            # retrieve the channel group ids
            channel_group_ids = station.get_channel_ids(return_group_ids=True)

            # iterate over each channel group ID
            for ch_grp_id in channel_group_ids:
                f_tot = 0
                f_tot_err = 0
                for channel_per_group in station.iter_channel_group(ch_grp_id):
                    f_tot += channel_per_group.get_parameter(chp.signal_fluence)
                    f_tot_err += channel_per_group.get_parameter_error(chp.signal_fluence)**2  # sum in quadrature
                f_tot_err = np.sqrt(f_tot_err)  # take the square root to get the total error

                data_fluences.append(f_tot)
                data_fluences_err.append(f_tot_err)
                
                data_ant_pos.append(det.get_absolute_position(station.get_id()) + det.get_relative_position(station.get_id(), channel_per_group.get_id()))
                
        # convert to numpy arrays
        data_fluences = np.array(data_fluences)
        data_fluences_err = np.array(data_fluences_err)
        data_ant_pos = np.array(data_ant_pos)
        
        print("data fluence max:", np.max(data_fluences), "data fluence err max:", np.max(data_fluences_err))

        return data_fluences, data_fluences_err, data_ant_pos

    @staticmethod
    def _get_or_create_radio_shower(event):
        showers = list(event.get_showers())
        if showers:
            return showers[0]
        import NuRadioReco.framework.radio_shower

        shower = NuRadioReco.framework.radio_shower.RadioShower(
            shower_id=event.get_id(), station_ids=event.get_station_ids()
        )
        event.add_shower(shower)
        return shower

    @register_run()
    def run(self, data_event, det, selected_station_channel_ids=None):
        """
        Run the reconstruction module.
        
        This performs the following:
        1. Calculate the fluence of the data event per station.
        2. For each simulation in the input files:
            a. Read the event from the input files.
            b. Apply the signal chain to the simulated event (Efield -> Voltage -> Bandpass -> Resample -> Signal Properties).
            c. Calculate the fluence of the simulated event per station.
            d. Evaluate the chi2 between the data fluence and the simulated fluence.
        3. Fit a parabola to the lower envelope of the chi2 vs Xmax values to find the best fit Xmax.

        Parameters
        ----------
        data_event : NuRadioReco.framework.event.Event
            The event to be processed. This is the *data* event, not the *sim* event.
        det : NuRadioReco.detector.Detector
            The detector to be used for the reconstruction. This should be same as the detector used for the simulation.
        selected_station_channel_ids : list of int, optional
            The list of channel IDs to be processed. If None, all channels will be processed.
        """
        # read the event from the input files
        if self.input_files is None:
            raise ValueError(
                "Input files are not set. Please set the input files using the 'begin' method."
            )
        if len(self.input_files) == 0:
            raise ValueError("Input files are empty. Please provide valid input files.")

        n_sims = len(self.input_files)

        reco_red_chi2_vals = np.full(n_sims, np.nan)
        reco_xmaxs = np.full(n_sims, np.nan)
        reco_core_x_fit = np.full(n_sims, np.nan)
        reco_core_y_fit = np.full(n_sims, np.nan)
        reco_scale_factor_sq_fit = np.full(n_sims, np.nan)
        reco_successes = np.zeros(n_sims, dtype=bool)

        # first compute the data fluence
        data_fluences, data_fluences_err, data_ant_pos = self._get_data_fluences(
            data_event, det, selected_station_channel_ids
        )

        # get the MC xmax if the data is a simulated event
        mc_xmax = None
        mc_core = None
        try:
            mc_xmax = data_event.get_first_sim_shower().get_parameter(shp.shower_maximum)
            mc_core = data_event.get_first_sim_shower().get_parameter(shp.core)
        except Exception:
            mc_xmax = None
            mc_core = None
            
        # the core guess is set from the estimates from the particle detectors, if they exist. Otherwise, we use the core from the data event, which is usually at (0, 0) for CoREAS simulations.
        try:
            hybrid_shower = data_event.get_hybrid_information().get_hybrid_shower(self.__settings["hybrid_shower_name"])
            core_guess = hybrid_shower.get_parameter(shp.core)[:2]  # only take the x and y coordinates
        except Exception:
            core_guess = np.array([0.0, 0.0])  # default to (0, 0) if no hybrid shower information is available
            
        # initialise the miniChannelFluenceCalculator with the detector and data antenna positions
        self.miniChannelFluenceCalculator.begin(det, data_ant_pos, signal_window_length=self.__settings["fluence_window"], selected_station_channel_ids=selected_station_channel_ids)

        # now iterate over each simulation, read the event, apply the signal chain, calculate the fluence, and perform the fit
        for i, coreas_hdf5_file in enumerate(self.input_files):

            corsika_sim_event = self._read_event(coreas_hdf5_file)

            fit_result = self._perform_fluence_fit(
                corsika_sim_event,
                det,
                data_fluences,
                data_fluences_err,
                core_guess,
                selected_station_channel_ids,
            )

            reco_red_chi2_vals[i] = fit_result["chi2"] / fit_result["n_dof"]
            reco_xmaxs[i] = corsika_sim_event.get_first_sim_shower().get_parameter(shp.shower_maximum) # in internal units
            reco_core_x_fit[i] = fit_result["core_x"]
            reco_core_y_fit[i] = fit_result["core_y"]
            reco_scale_factor_sq_fit[i] = fit_result["scale_factor_sq"]
            reco_successes[i] = fit_result["success"]

        self.logger.info(
            f"Fluence fit completed for {n_sims} simulations. Successful fits: {np.sum(reco_successes)}"
        )

        # only consider successful fits for the parabola fit
        reco_red_chi2_vals_successful = reco_red_chi2_vals[reco_successes]
        reco_xmaxs_successful = reco_xmaxs[reco_successes]
        reco_core_x_fit_successful = reco_core_x_fit[reco_successes]
        reco_core_y_fit_successful = reco_core_y_fit[reco_successes]
        reco_scale_factor_sq_fit_successful = reco_scale_factor_sq_fit[reco_successes]
        
        # plot the reconstructed histogram from all successful fits
        try:
            fig = plot_reco_histograms(
                reco_red_chi2_vals_successful,
                reco_core_x_fit_successful,
                reco_core_y_fit_successful,
                reco_scale_factor_sq_fit_successful,
                mc_core_x=mc_core[0] if mc_core is not None else None,
                mc_core_y=mc_core[1] if mc_core is not None else None,
                save_path=os.path.join(
                    self.__settings["output_directory"],
                    f"reco_histograms_event_{data_event.get_id()}.png",
                )
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to plot reconstruction histograms. Error: {e}", exc_info=True
            )

        # now fit a parabola to the lower envelope of the chi2 vs Xmax values to find the best fit Xmax
        (
            parabola_fit_param,
            xmax_reconstructed,
            fit_converged,
        ) = fit_to_lower_envelope_parabola(
            reco_xmaxs_successful,
            reco_red_chi2_vals_successful,
            range_around_best_xmax=self.__settings["xmax_parabola_fit_range"],
        )
        
        lower_envelope_indices = get_lower_envelope_indices(
            reco_xmaxs_successful,
            reco_red_chi2_vals_successful,
            range_around_best_xmax=self.__settings["xmax_parabola_fit_range"],
        )

        self.logger.info(
            f"Parabola fit succeeded: {fit_converged}."
        )
        if fit_converged:
            self.logger.info(
                f"Parabola fit parameters: {parabola_fit_param[0]:.2f} x^2 + {parabola_fit_param[1]:.2f} x + {parabola_fit_param[2]:.2f}, \n"
                f"Chi2 at minimum: {parabola_fit_param[2]:.2f}, \n"
                f"Reconstructed Xmax: {xmax_reconstructed / (units.g / units.cm**2):.2f} g/cm^2", 
            )

        # plot the chi2 parabola
        for reco_fit_param, label in zip(
            [reco_scale_factor_sq_fit_successful, reco_core_x_fit_successful, reco_core_y_fit_successful],
            ["scale_factor", "core_x", "core_y"],
        ):
            if label == "scale_factor":
                # take the square root of the scale factor squared to get the scale factor
                reco_fit_param = np.sqrt(reco_fit_param)

            try:
                fig = plot_chi2_parabola(
                    reco_xmaxs_successful,
                    reco_red_chi2_vals_successful,
                    reco_fit_param,
                    parabola_fit_param,
                    xmax_reconstructed,
                    lower_envelope_indices=lower_envelope_indices,
                    xmax_fit_window=self.__settings["xmax_parabola_fit_range"],
                    xmax_inset_window=self.__settings["xmax_parabola_fit_range"],
                    mc_xmax=mc_xmax,
                    fit_param_label=label,
                    parab_color=self.__settings["parabola_color"],
                    save_path=os.path.join(
                        self.__settings["output_directory"],
                        f"chi2_parabola_{label}_event_{data_event.get_id()}.png",
                    )
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to plot chi2 parabola for fit parameter {label}. Error: {e}", exc_info=True
                )


    def end(self):
        """
        Finalize the module.
        """
        pass