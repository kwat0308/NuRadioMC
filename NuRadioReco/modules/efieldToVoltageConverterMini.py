import os
import glob
import h5py
import logging
import argparse
import numpy as np
import radiotools
import json
import time
import functools

from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy import constants

from NuRadioReco.framework import station
import NuRadioReco.framework.event
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.modules.io.coreas import coreasInterpolator, coreas
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

from NuRadioReco.modules.efieldToVoltageConverter import efieldToVoltageConverter, calculate_time_shift_for_cosmic_ray
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import channelParameters as chp

from NuRadioReco.utilities import units

class efieldToVoltageConverterMini(efieldToVoltageConverter):
    """
    Convert electric fields to voltages for a given station and detector.

    This module takes the simulated electric fields from a station and converts them to voltages using the antenna response. The resulting voltages are stored in the station's channels.

    Parameters
    ----------
    det : NuRadioReco.framework.detector.Detector
        The detector object containing the antenna response information.
    antenna_provider : NuRadioReco.modules.antenna.AntennaProvider
        The antenna provider used to obtain the antenna response.

    Methods
    -------
    convert(station)
        Convert the electric fields of the given station to voltages.
    """

    def __init__(self, log_level=logging.DEBUG):
        self.__t = 0
        self.__t_antenna_pattern = 0
        self.__t_antenna_sensitivity = 0
        self.__t_time_shift = 0
        self.__t_channel_properties = 0
        self.__t_fluence = 0
        self.__freqs = None
        self.logger = logging.getLogger('NuRadioReco.efieldToVoltageConverterMini')
        self.logger.setLevel(log_level)

        super().__init__(log_level=log_level)

    @functools.lru_cache(maxsize=1024)
    def _get_cached_antenna_response(self, ant_pattern, zen, azi, *ant_orient):
        """
        Returns the cached antenna reponse for a given antenna pattern, antenna orientation
        and signal arrival direction. This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return ant_pattern.get_antenna_response_vectorized(self.__freqs, zen, azi, *ant_orient)

    @register_run()
    def run(self, evt, station, det, channel_ids=None):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()
        if len(sim_station.get_electric_fields()) == 0:
            raise LookupError(f"station {station.get_id()} has no efields")
        
        channel_ids_to_use = det.get_channel_ids(sim_station_id) if channel_ids == None else channel_ids[station.get_id()]

        # loop over all channels
        for channel_id in channel_ids_to_use:

            # one channel might contain multiple channels to store the signals from multiple ray paths,
            # so we loop over all simulated channels with the same id,
            # convolve each trace with the antenna response for the given angles
            # and everything up in the time domain
            self.logger.debug('channel id {}'.format(channel_id))
            channel = NuRadioReco.framework.channel.Channel(channel_id, channel_group_id=det.get_channel_group_id(sim_station_id, channel_id))

            channel_spectrum = None
            for electric_field in sim_station.get_electric_fields_for_channels([channel_id]):

                ff = electric_field.get_frequencies()
                self.__freqs = ff
                efield_fft = electric_field.get_frequency_spectrum()

                zenith = electric_field[efp.zenith]
                azimuth = electric_field[efp.azimuth]

                # Get the antenna pattern and orientation for the current channel
                t_step = time.time()
                antenna_pattern, antenna_orientation = self.get_antenna_pattern_and_orientation(
                    det, sim_station, channel_id, zenith)
                self.__t_antenna_pattern += time.time() - t_step

                # Get antenna sensitivity for the given direction
                t_step = time.time()
                # vel = antenna_pattern.get_antenna_response_vectorized(
                #     ff, zenith, azimuth, *antenna_orientation)
                vel = self._get_cached_antenna_response(
                        antenna_pattern, zenith, azimuth, *antenna_orientation)
                self.__t_antenna_sensitivity += time.time() - t_step

                if vel is None:  # this can happen if there is not signal path to the antenna
                    voltage_fft = np.zeros_like(efield_fft[1])  # set voltage trace to zeros
                else:
                    # Apply antenna response to electric field
                    vel = np.array([vel['theta'], vel['phi']])
                    voltage_fft = np.sum(vel * np.array([efield_fft[1], efield_fft[2]]), axis=0)

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.

                t_step = time.time()
                dist_channel_efield = np.linalg.norm(det.get_relative_position(sim_station.get_id(), channel_id) - electric_field.get_position())
                travel_time_shift = calculate_time_shift_for_cosmic_ray(
                    det, sim_station, electric_field, channel_id) if dist_channel_efield / units.mm > 0.01 else 0
                self.__t_time_shift += time.time() - t_step

                if channel_spectrum is None:
                    channel_spectrum = voltage_fft
                else:
                    channel_spectrum += voltage_fft

            t_step = time.time()
            channel.set_frequency_spectrum(channel_spectrum, electric_field.get_sampling_rate())

            channel.set_trace_start_time(electric_field.get_trace_start_time() + travel_time_shift)
            self.__t_channel_properties += time.time() - t_step

            station.add_channel(channel)

        self.__t += time.time() - t
        self.logger.debug(
            f"timing (s): antenna_pattern={self.__t_antenna_pattern:.4f}, "
            f"antenna_sensitivity={self.__t_antenna_sensitivity:.4f}, "
            f"time_shift={self.__t_time_shift:.4f}, "
            f"channel_properties={self.__t_channel_properties:.4f}, "
            f"fluence={self.__t_fluence:.4f}, total={self.__t:.4f}"
        )
