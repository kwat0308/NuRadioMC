#!/usr/bin/env python3
"""
NuRadioReco-native LOFAR data pipeline.

The base LOFAR pipeline takes a LOFAR event ID and a detector description JSON file as input, and produces a .nur file containing the reconstructed shower parameters and station parameters. The pipeline also produces debug plots if requested.

This pipeline performs the fluence reconstruction for a single data event using the CoREAS simulation corresponding to the given event ID. The pipeline performs the following steps:
1. Generate a NuRadioReco event from the CoREAS simulation using the CoREASEventGenerator module.
2. Convert the voltage traces to electric field traces using the voltageToEfieldConverter module.
3. Run the fluence reconstruction using the CoREASFluenceReconstructor module.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de> & Keito Watanabe <keito.watanabe@kit.edu>
"""

import argparse
import logging
import os

import numpy as np  # noqa: E402

import NuRadioReco.detector.detector  # noqa: E402
from NuRadioReco.framework.parameters import (
    stationParameters,
    showerParameters,
)  # noqa: E402
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.modules.LOFAR.event_gen.CoREASEventGenerator import (
    CoREASEventGenerator,
)  # noqa: E402
from NuRadioReco.modules import voltageToEfieldConverter  # noqa: E402
from NuRadioReco.modules.LOFAR import pipelineVisualizer_LOFAR  # noqa: E402
import NuRadioReco.modules.io.eventWriter  # noqa: E402
from NuRadioReco.modules.LOFAR.reconstruction.CoREASFluenceReconstructor import (
    CoREASFluenceReconstructor,
)  # noqa: E402
from NuRadioReco.utilities.LOFAR.constants import (  # noqa: E402
    GDAS_ATMOSPHERE_DIRECTORY,
    COREAS_PARENT_HDF5_DIRECTORY,
    SIM_CORE_SPREAD,
    PASS_BAND,
    BANDPASS_HALF_HANN_PERCENT,
    BANDPASS_GAUSSIAN_ROLL_WIDTH,
)
from NuRadioReco.utilities import units  # noqa: E402
from NuRadioReco.utilities.LOFAR.fluence_reco_utilities import make_xmax_bias_plot  # noqa: E402

LOGGER = logging.getLogger("NuRadioReco.pipeline.LOFAR.fluence_reco_pipeline")

_SIGNAL_FLUENCE_WINDOW = 55.0  # in ns
_XMAX_PARABOLA_WINDOW = 100.0  # in g/cm2


def _convert_voltage_to_efield(event, detector):
    converter = voltageToEfieldConverter.voltageToEfieldConverter()
    converter.begin()

    for station in event.get_stations():
        if not station.has_parameter(
            stationParameters.zenith
        ) or not station.has_parameter(stationParameters.azimuth):
            shower = next(event.get_showers())
            if shower.has_parameter(showerParameters.zenith) and shower.has_parameter(
                showerParameters.azimuth
            ):
                station.set_parameter(
                    stationParameters.zenith,
                    shower.get_parameter(showerParameters.zenith),
                )
                station.set_parameter(
                    stationParameters.azimuth,
                    shower.get_parameter(showerParameters.azimuth),
                )
            else:
                station.set_parameter(stationParameters.zenith, 0.0 * units.radian)
                station.set_parameter(stationParameters.azimuth, 0.0 * units.radian)

        for group_id in station.get_channel_ids(return_group_ids=True):
            use_channels = [
                channel.get_id() for channel in station.iter_channel_group(group_id)
            ]
            if len(use_channels) < 2:
                continue
            try:
                converter.run(
                    event,
                    station,
                    detector,
                    use_channels=use_channels,
                    use_MC_direction=False,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Skipping e-field conversion for station %s group %s: %s",
                    station.get_id(),
                    group_id,
                    exc,
                )

    converter.end()


def _write_params_nur(event, output_dir):
    """Write a .nur file containing only shower/station parameters (no channel traces or e-fields)."""
    for station in event.get_stations():
        for ch_id in list(station.get_channel_ids()):
            station.remove_channel(ch_id)
        station.set_electric_fields([])
    path = os.path.join(output_dir, f"{event.get_id()}.nur")
    writer = NuRadioReco.modules.io.eventWriter.eventWriter()
    writer.begin(path)
    writer.run(event)
    writer.end()
    LOGGER.info("Wrote parameters-only .nur to %s", path)


def _save_pipeline_visualizer_plots(event, detector, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    visualizer = pipelineVisualizer_LOFAR.pipelineVisualizer()
    visualizer.begin()
    try:
        visualizer.run(
            event, detector, save_dir=output_dir, polarization=True, direction=True
        )
    except Exception as exc:
        LOGGER.warning("Skipping final LOFAR visualizer debug plots: %s", exc)
    finally:
        visualizer.end()
        
def get_xmax_from_input_files(input_files):
    """
    Get the Xmax value from the input files. This is used to set the initial guess for the Xmax fit.
    """
    xmax_values = []
    for input_file in input_files:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        corsika_evt = coreas.read_CORSIKA7(input_file)
        xmax_values.append(
            corsika_evt.get_first_sim_shower().get_parameter(showerParameters.shower_maximum) / (units.g / units.cm**2)
        )
        
    return xmax_values

def run_pipeline(args):

    # manually generate detector with manual json file
    detector = NuRadioReco.detector.detector.Detector(
        args.detector,
        source="json",
        antenna_by_depth=False,
    )
    
    # we iterate over the events in the CoREAS HDF5 file, since the fluence reconstruction in the simulation side is performed by making each simulation into data.
    # this way we can also quantify the uncertainty in xmax directly.
    LOGGER.info(
                "Running CoREAS Fluence Reconstruction module for event %d",
                args.event_id)
    
    # first get all input files
    coreas_event_path = os.path.join(args.coreas_dir, str(args.event_id))
    
    data_coreas_files = []
    for root, dirs, files in os.walk(coreas_event_path):
        for file in files:
            if file.endswith(".hdf5"):
                data_coreas_files.append(os.path.join(root, file))
    true_xmaxs = get_xmax_from_input_files(data_coreas_files)
    # sort based on true xmax
    data_coreas_files = [x for _, x in sorted(zip(true_xmaxs, data_coreas_files))]
    true_xmaxs = np.array(sorted(true_xmaxs))

    # now remove the last 5 and first 5 showers from the sorted list
    true_xmaxs = true_xmaxs[5:-5]
    data_coreas_files = data_coreas_files[5:-5]
    n_showers_data = len(true_xmaxs)
    LOGGER.info(f"After removing outliers, {n_showers_data} showers remain for full reconstruction of event {str(args.event_id)}.")

    xmax_reconstructed = []
    for data_file_path in data_coreas_files:
        
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(
                f"CoREAS HDF5 file not found or is not a file: {data_file_path}"
            )
        
        # for naming convention, get the mass & run number from the path of the CoREAS HDF5 file
        run_number = os.path.basename(os.path.dirname(data_file_path)).replace(".hdf5", "")
        mass = os.path.basename(os.path.dirname(os.path.dirname(data_file_path)))
        
        LOGGER.info(
            f"Taking CoREAS HDF5 file {data_file_path} as input for event {args.event_id}, mass {mass}, run {run_number}",
        )

        # the directory structure is such that the output directory will have a subdirectory for each event ID.
        # For now we will also create a subdirectory for each mass, since it is unclear whether there is overlap in the simulation run numbers for different masses. In the future, we should be able to remove the mass subdirectory and just use the event ID subdirectory.
        # then in each subdirectory, the nur files (and corresponding debug plots) will be written with the name of the CoREAS HDF5 file (without the .hdf5 extension and the "SIM" prefix) as the file name.

        event_output_dir = os.path.join(args.output_dir, str(args.event_id), mass, run_number)
        os.makedirs(event_output_dir, exist_ok=True)
        LOGGER.info(
            "Output directory for event %d, mass %s, simulation run %s: %s",
            args.event_id,
            mass,
            run_number,
            event_output_dir,
        )

        coreas_event_generator = CoREASEventGenerator(
            detector,
            output_directory=event_output_dir,
            noise_library_file=args.noise_library,
            noise_library_nur_file=args.noise_library_nur,
            antenna_mode=args.antenna_mode,
            core_spread_m=args.core_spread,
            atmosphere_dir=args.atmosphere_dir,
            gdas_cache_dir=args.gdas_cache_dir,
        )

        processed_event = coreas_event_generator.process_event(
            coreas_hdf5_file=data_file_path,
            save_debug_plots=args.debug_plots,
            write_event=False,
        )

        # for now fixed, in the future we should be able to replace this
        reconstructor = CoREASFluenceReconstructor()

        LOGGER.info("Running voltage-to-electric-field conversion")
        _convert_voltage_to_efield(processed_event, detector)
        if args.debug_plots:
            _save_pipeline_visualizer_plots(
                processed_event, detector, coreas_event_generator.debug_dir
            )

        LOGGER.info("Running CoREAS Fluence Reconstruction module")

        # find the input files based on the parent HDF5 file and the event ID
        # but exclude the CoREAS HDF5 file that was used to generate the event.
        coreas_input_files = []
        for root, dirs, files in os.walk(coreas_event_path):
            for file in files:
                coreas_hdf5_path = os.path.join(root, file)
                if file.endswith(".hdf5") and coreas_hdf5_path != data_file_path:
                    coreas_input_files.append(os.path.join(root, file))

        recon_kwargs = dict(
            input_files=coreas_input_files,
            site="lofar",
            fluence_window_ns=args.signal_fluence_window,
            pass_band_MHz=PASS_BAND,
            bandpass_gaussian_roll_width_MHz=BANDPASS_GAUSSIAN_ROLL_WIDTH,
            bandpass_half_hann_percent=BANDPASS_HALF_HANN_PERCENT,
            xmax_parabola_fit_range_gcm2=args.xmax_parabola_window,
            output_directory=event_output_dir,
            no_core_fit=args.no_core_fit,
            hybrid_shower_name="LORA",
            logger_level = logging.INFO,
        )

        reconstructor.begin(**recon_kwargs)
        processed_event = reconstructor.run(
            processed_event,
            detector,
            selected_station_channel_ids=coreas_event_generator.selected_station_channel_ids,
        )
        reconstructor.end()

        if args.output_nur is not None:
            writer = NuRadioReco.modules.io.eventWriter.eventWriter()
            writer.begin(args.output_nur)
            writer.run(processed_event)
            writer.end()

        os.makedirs(event_output_dir, exist_ok=True)
        _write_params_nur(processed_event, event_output_dir)
        
        reconstructed_xmax = processed_event.get_first_sim_shower().get_parameter(showerParameters.shower_maximum) / (units.g / units.cm**2)
        xmax_reconstructed.append(reconstructed_xmax)
        LOGGER.info(
            "Finished processing event %d, mass %s, simulation run %s. Reconstructed Xmax: %.2f g/cm2, True Xmax: %.2f g/cm2",
            args.event_id,
            mass,
            run_number,
            reconstructed_xmax,
            true_xmaxs[data_coreas_files.index(data_file_path)],
        )
        
    print(f"Finished processing event {args.event_id}. Reconstructed Xmax values: {xmax_reconstructed}, True Xmax values: {true_xmaxs.tolist()}")
    
    if args.debug_plots:
        bias = np.array(xmax_reconstructed) - true_xmaxs
        mean_bias, resolution = make_xmax_bias_plot(bias, "Xmax", "blue", os.path.join(args.output_dir, f"xmax_bias_event_{args.event_id}.png"), f"Event {args.event_id}")
        LOGGER.info("Saved Xmax bias plot to %s", os.path.join(args.output_dir, f"xmax_bias_event_{args.event_id}.png"))
        LOGGER.info("Mean bias: %.2f g/cm2, Resolution: %.2f g/cm2", mean_bias, resolution)
        


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "event_id",
        type=int,
        help="event ID used to identify the LOFAR event to process.",
    )

    parser.add_argument(
        "--detector",
        type=str,
        help="Path to the detector description JSON file",
        default="LOFAR/LOFAR.json",
    )

    parser.add_argument(
        "--signal-fluence-window",
        type=float,
        default=_SIGNAL_FLUENCE_WINDOW,
        help="Time window in ns for the signal fluence calculation (default: %(default)s ns).",
    )

    parser.add_argument(
        "--xmax-parabola-window",
        type=float,
        default=_XMAX_PARABOLA_WINDOW,
        help="Fit range in g/cm2 for the Xmax parabola fit (default: %(default)s g/cm2).",
    )
    
    parser.add_argument(
        "--no-core-fit",
        action="store_true",
        help="If set, the core position will not be fitted and will be fixed to the core position of the data event. If not set, the core position will be fitted along with the scale factor.",
    )

    parser.add_argument(
        "--output-dir",
        default=os.getcwd(),
        help="Directory for output files and debug plots",
    )
    parser.add_argument(
        "--output-nur", default=None, help="Write the processed event to this .nur file"
    )

    parser.add_argument(
        "--coreas-dir",
        default=COREAS_PARENT_HDF5_DIRECTORY,
        help="Root directory holding the per-event CoREAS HDF5 trees, laid out as "
        "<coreas-dir>/<event_id>/<run>/<mass>/<coreas_hdf5_file> (default: %(default)s).",
    )
    parser.add_argument(
        "--noise-library",
        default=None,
        help="LOFAR measured-noise library .npy (default: the macros' path).",
    )
    parser.add_argument(
        "--noise-library-nur",
        default=None,
        help="Where to write the .nur conversion of the noise library. Must be "
        "writable (default: the macros' path).",
    )
    parser.add_argument(
        "--atmosphere-dir",
        default=GDAS_ATMOSPHERE_DIRECTORY,
        help="Directory to search for pre-computed ATMOSPHERE_{event_id}.DAT files "
        "(may be read-only). If a file is not found it is generated and written "
        "to --gdas-cache-dir.",
    )
    parser.add_argument(
        "--gdas-cache-dir",
        default=os.path.join(os.path.expanduser("~"), ".cache", "lofar_gdas"),
        help="Writable directory for downloaded GDAS binaries and newly generated "
        "ATMOSPHERE_*.DAT files. Defaults to ~/.cache/lofar_gdas.",
    )

    parser.add_argument(
        "--antenna-mode",
        choices=("lba_all", "lba_outer", "lba_inner"),
        default="lba_all",
        help="Which LBA set to simulate and reconstruct. LOFAR1.0 reads out one mode "
        "at a time, LOFAR2.0 hopefully can read both simultaneously, "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--core-spread",
        type=float,
        default=SIM_CORE_SPREAD,
        help="Standard deviation, per horizontal coordinate, of the Gaussian the "
        "physical shower core is drawn from. The LORA core guess is this core "
        "displaced by the LORA resolution (default: %(default)s m).",
    )
    parser.add_argument("--debug-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    event = run_pipeline(args)
    if event is None:
        raise RuntimeError(f"Pipeline did not produce event {args.event_id}")
    LOGGER.info(
        "Finished event %s with %d stations",
        event.get_id(),
        len(event.get_station_ids()),
    )


if __name__ == "__main__":
    main()
