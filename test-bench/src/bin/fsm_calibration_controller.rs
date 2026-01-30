//! FSM Calibration Controller
//!
//! Calibrates the relationship between FSM (Fast Steering Mirror) commands
//! and centroid motion on the sensor. Uses static step calibration with
//! settle time between positions.
//!
//! The controller connects directly to the PI E-727 controller via ethernet
//! and receives centroid positions via SSE from fgs_server.

use clap::Parser;
use hardware::pi::S330;
use shared_wasm::{CalibrateServerClient, FgsServerClient};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use test_bench::fsm_calibration::{
    CalibrationRawData, FsmAxisCalibration, FsmCalibrationConfig, StaticCalibrationError,
    StaticStepExecutor,
};
use test_bench::tracking_collector::TrackingCollector;
use tracing::{error, info, warn};

/// FSM Calibration Controller
///
/// Calibrates the FSM-to-sensor transform by stepping to discrete positions
/// and measuring the resulting centroid displacement.
#[derive(Parser, Debug)]
#[command(name = "fsm_calibration_controller")]
#[command(
    about = "Calibrate FSM-to-sensor transform via static step positions",
    long_about = "FSM Calibration Controller for determining the relationship between \
        Fast Steering Mirror (FSM) commands and centroid motion on the sensor.\n\n\
        The calibration process:\n  \
        1. Connects directly to the PI E-727 controller via ethernet\n  \
        2. Steps axis 1 to -amplitude, 0, +amplitude positions\n  \
        3. Waits for settle time, then collects tracking samples at each position\n  \
        4. Repeats for axis 2\n  \
        5. Computes response vectors from centroid displacement\n  \
        6. Builds 2x2 transform matrix mapping FSM µrad to sensor pixels\n\n\
        Prerequisites:\n  \
        - PI E-727 controller accessible on the network\n  \
        - fgs_server running and tracking a stable star/spot"
)]
struct Args {
    #[arg(
        long,
        default_value = "192.168.15.201",
        help = "PI E-727 controller IP address",
        long_help = "IP address of the PI E-727 piezo controller for the S-330 Fast Steering \
            Mirror. The controller connects directly via ethernet on GCS port 50000."
    )]
    fsm_ip: String,

    #[arg(
        long,
        help = "Keep FSM powered on after calibration",
        long_help = "By default, the S-330 driver disables servos on drop for safety. \
            Set this flag to keep the FSM powered and holding position after calibration."
    )]
    keep_powered: bool,

    #[arg(
        long,
        default_value = "http://cfl-test-bench.tail944341.ts.net",
        help = "URL of calibrate_serve for display control",
        long_help = "URL of calibrate_serve instance for setting the calibration pattern \
            on the OLED display. Format: http://host"
    )]
    calibrate_serve_url: String,

    #[arg(
        long,
        default_value = "http://orin-005.tail944341.ts.net:3000",
        help = "Base URL of fgs_server for tracking control",
        long_help = "Base URL of fgs_server for controlling tracking enable/disable \
            during calibration. Format: http://host:port"
    )]
    fgs_server_url: String,

    #[arg(
        long,
        help = "Output CSV file for raw calibration data",
        long_help = "If provided, writes all raw sample data to a CSV file for analysis. \
            The CSV contains: axis, fsm_command_urad, sample_index, centroid_x, centroid_y, timestamp_s"
    )]
    output_csv: Option<PathBuf>,

    /// Calibration parameters (swing range, num steps, samples, etc.)
    #[command(flatten)]
    calibration: FsmCalibrationConfig,
}

/// Connect to FSM and configure power-off behavior
fn connect_fsm(ip: &str, keep_powered: bool) -> Result<S330, String> {
    let mut fsm =
        S330::connect_ip(ip).map_err(|e| format!("Failed to connect to FSM at {ip}: {e}"))?;

    if keep_powered {
        fsm.set_poweroff_on_drop(false);
    }

    Ok(fsm)
}

/// Write raw calibration data to CSV file
fn write_csv(
    path: &PathBuf,
    raw_data: &CalibrationRawData,
    calibration: &FsmAxisCalibration,
) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Failed to create CSV file: {e}"))?;

    // Write header with calibration metadata as comments
    writeln!(file, "# FSM Calibration Raw Data")
        .map_err(|e| format!("Failed to write CSV: {e}"))?;
    writeln!(
        file,
        "# Intercept: ({:.4}, {:.4}) pixels",
        calibration.intercept_pixels.x, calibration.intercept_pixels.y
    )
    .map_err(|e| format!("Failed to write CSV: {e}"))?;
    writeln!(
        file,
        "# FSM-to-Sensor: [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
        calibration.fsm_to_sensor[(0, 0)],
        calibration.fsm_to_sensor[(0, 1)],
        calibration.fsm_to_sensor[(1, 0)],
        calibration.fsm_to_sensor[(1, 1)]
    )
    .map_err(|e| format!("Failed to write CSV: {e}"))?;
    writeln!(
        file,
        "# Sensor-to-FSM: [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
        calibration.sensor_to_fsm[(0, 0)],
        calibration.sensor_to_fsm[(0, 1)],
        calibration.sensor_to_fsm[(1, 0)],
        calibration.sensor_to_fsm[(1, 1)]
    )
    .map_err(|e| format!("Failed to write CSV: {e}"))?;
    writeln!(
        file,
        "# Axis 1 std: {:.4} pixels, Axis 2 std: {:.4} pixels",
        calibration.axis1_std_pixels, calibration.axis2_std_pixels
    )
    .map_err(|e| format!("Failed to write CSV: {e}"))?;

    // Write CSV header
    writeln!(
        file,
        "axis,fsm_command_urad,sample_index,centroid_x,centroid_y,timestamp_s"
    )
    .map_err(|e| format!("Failed to write CSV: {e}"))?;

    // Write data rows
    for sample in &raw_data.samples {
        writeln!(
            file,
            "{},{:.4},{},{:.4},{:.4},{:.6}",
            sample.axis,
            sample.fsm_command_urad,
            sample.sample_index,
            sample.centroid_x,
            sample.centroid_y,
            sample.timestamp_s
        )
        .map_err(|e| format!("Failed to write CSV: {e}"))?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("FSM Calibration Controller");
    info!("==========================");
    info!("FSM IP:            {}", args.fsm_ip);
    info!("FGS server:        {}", args.fgs_server_url);
    info!(
        "Swing range:       {:.1} µrad",
        args.calibration.swing_range_urad
    );
    info!("Num steps:         {}", args.calibration.num_steps);
    info!(
        "Lock wait timeout: {:.1} s",
        args.calibration.lock_on_time_secs
    );
    info!(
        "Samples/position:  {}",
        args.calibration.samples_per_position
    );
    info!("Discard samples:   {}", args.calibration.discard_samples);
    info!("Keep powered:      {}", args.keep_powered);
    info!("Calibrate serve:   {}", args.calibrate_serve_url);
    if let Some(ref path) = args.output_csv {
        info!("Output CSV:        {:?}", path);
    }
    info!("");

    // Create FGS client for tracking control
    let fgs_client = FgsServerClient::new(&args.fgs_server_url);

    // Create calibrate client and set up display pattern
    let calibrate_client = CalibrateServerClient::new(&args.calibrate_serve_url);

    // Set display to center pixel pattern
    info!("Setting display to center pixel pattern...");
    calibrate_client.set_pixel_pattern().await.map_err(|e| {
        error!("Failed to set pixel pattern on calibrate_serve: {e}");
        e.to_string()
    })?;

    // Enable tracking after setting the pattern
    info!("Enabling tracking...");
    fgs_client.set_tracking_enabled(true).await.map_err(|e| {
        error!("Failed to enable tracking: {e}");
        e.to_string()
    })?;

    // Get display info for logging
    let display_info = calibrate_client.get_display_info().await.map_err(|e| {
        error!("Failed to get display info from calibrate_serve: {e}");
        e.to_string()
    })?;

    if let Some(pitch) = display_info.pixel_pitch_um {
        info!(
            "Display: {}x{} pixels, {:.1} µm pitch",
            display_info.width, display_info.height, pitch
        );
    } else {
        info!(
            "Display: {}x{} pixels",
            display_info.width, display_info.height
        )
    };

    // Connect to FSM via direct ethernet
    info!("Connecting to FSM at {}...", args.fsm_ip);
    let mut fsm = match connect_fsm(&args.fsm_ip, args.keep_powered) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to connect to FSM: {e}");
            return Err(e.into());
        }
    };

    // Get travel ranges and determine center position (always auto-detected)
    let center_position_urad = match fsm.get_travel_ranges() {
        Ok((x_range, y_range)) => {
            info!(
                "FSM travel ranges: X=[{:.1}, {:.1}], Y=[{:.1}, {:.1}] µrad",
                x_range.0, x_range.1, y_range.0, y_range.1
            );
            let center = (x_range.0 + x_range.1) / 2.0;
            info!("Center position:   {:.1} µrad (auto-detected)", center);
            center
        }
        Err(e) => {
            error!("Could not get FSM travel ranges: {e}");
            return Err(format!("FSM travel range error: {e}").into());
        }
    };

    // Connect to tracking stream using the tracking events URL from fgs_client
    let tracking_events_url = fgs_client.tracking_events_url();
    info!(
        "Connecting to tracking stream at {}...",
        tracking_events_url
    );
    let collector = match TrackingCollector::connect(&tracking_events_url) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to connect to tracking: {e}");
            return Err(e.into());
        }
    };

    // Wait for tracking messages
    info!(
        "Waiting for tracking messages ({:.1}s timeout)...",
        args.calibration.lock_on_time_secs
    );
    match collector.wait_for_message(Duration::from_secs_f64(args.calibration.lock_on_time_secs)) {
        Ok(true) => {}
        Ok(false) => {
            error!(
                "No tracking messages received within {:.1} seconds",
                args.calibration.lock_on_time_secs
            );
            error!("Make sure fgs_server is running and tracking a spot");
            return Err("No tracking messages".into());
        }
        Err(e) => {
            error!("Tracking connection error: {}", e);
            return Err("Tracking connection lost".into());
        }
    }
    info!("Tracking stream active");

    // Set auto-detected center on the config
    let mut config = args.calibration;
    config.center_position_urad = center_position_urad;

    info!("");

    // Run calibration
    let mut executor =
        StaticStepExecutor::new(fsm, collector, config, fgs_client, calibrate_client.clone());
    let calibration_result = executor.run_calibration().await;

    // Turn off display when calibration finishes (success or failure)
    info!("Turning off display...");
    if let Err(e) = calibrate_client.set_blank().await {
        warn!("Failed to turn off display: {e}");
    }

    match calibration_result {
        Ok((calibration, raw_data)) => {
            info!("");
            info!("=== Calibration Results ===");
            info!("");
            info!(
                "Intercept (centroid at FSM zero): ({:.2}, {:.2}) pixels",
                calibration.intercept_pixels.x, calibration.intercept_pixels.y
            );
            info!(
                "Axis 1 centroid std: {:.4} pixels",
                calibration.axis1_std_pixels
            );
            info!(
                "Axis 2 centroid std: {:.4} pixels",
                calibration.axis2_std_pixels
            );
            info!("");
            info!("FSM-to-Sensor Transform (pixels per µrad):");
            info!(
                "  [{:+.6}, {:+.6}]",
                calibration.fsm_to_sensor[(0, 0)],
                calibration.fsm_to_sensor[(0, 1)]
            );
            info!(
                "  [{:+.6}, {:+.6}]",
                calibration.fsm_to_sensor[(1, 0)],
                calibration.fsm_to_sensor[(1, 1)]
            );
            info!("");
            info!("Sensor-to-FSM Transform (µrad per pixel):");
            info!(
                "  [{:+.6}, {:+.6}]",
                calibration.sensor_to_fsm[(0, 0)],
                calibration.sensor_to_fsm[(0, 1)]
            );
            info!(
                "  [{:+.6}, {:+.6}]",
                calibration.sensor_to_fsm[(1, 0)],
                calibration.sensor_to_fsm[(1, 1)]
            );

            // Calculate scale and rotation
            let scale_x = (calibration.fsm_to_sensor[(0, 0)].powi(2)
                + calibration.fsm_to_sensor[(1, 0)].powi(2))
            .sqrt();
            let scale_y = (calibration.fsm_to_sensor[(0, 1)].powi(2)
                + calibration.fsm_to_sensor[(1, 1)].powi(2))
            .sqrt();
            let rotation =
                calibration.fsm_to_sensor[(1, 0)].atan2(calibration.fsm_to_sensor[(0, 0)]);

            info!("");
            info!("Derived parameters:");
            info!("  Scale X: {:.4} pixels/µrad", scale_x);
            info!("  Scale Y: {:.4} pixels/µrad", scale_y);
            info!("  Rotation: {:.2}°", rotation.to_degrees());

            // Write CSV if requested
            if let Some(ref csv_path) = args.output_csv {
                info!("");
                info!("Writing raw data to {:?}...", csv_path);
                match write_csv(csv_path, &raw_data, &calibration) {
                    Ok(()) => info!(
                        "CSV written successfully ({} samples)",
                        raw_data.samples.len()
                    ),
                    Err(e) => {
                        error!("Failed to write CSV: {e}");
                        return Err(e.into());
                    }
                }
            }

            Ok(())
        }
        Err(StaticCalibrationError::FsmError(e)) => {
            error!("Calibration failed: FSM error: {e}");
            Err(format!("FSM error: {e}").into())
        }
        Err(StaticCalibrationError::CentroidError(e)) => {
            error!("Calibration failed: centroid source error: {e}");
            Err(format!("Centroid error: {e}").into())
        }
        Err(StaticCalibrationError::InsufficientSamples { got, need }) => {
            error!(
                "Calibration failed: insufficient samples (got {}, need {})",
                got, need
            );
            warn!("Suggestions:");
            warn!("  - Ensure FGS is tracking a stable star/spot");
            warn!("  - Increase step amplitude (--step-amplitude-urad)");
            Err("Insufficient samples".into())
        }
        Err(StaticCalibrationError::DegenerateAxes(e)) => {
            error!("Calibration failed: {e}");
            warn!("The FSM axes appear to be nearly parallel");
            warn!("This may indicate a hardware or mounting issue");
            Err("Degenerate axes".into())
        }
        Err(StaticCalibrationError::NoMotionDetected { axis }) => {
            error!("Calibration failed: no motion detected for axis {}", axis);
            warn!("Suggestions:");
            warn!("  - Ensure FGS is tracking a stable star/spot");
            warn!("  - Increase step amplitude (--step-amplitude-urad)");
            warn!("  - Check FSM is responding to commands");
            Err(format!("No motion detected for axis {axis}").into())
        }
        Err(StaticCalibrationError::TrackingError(e)) => {
            error!("Calibration failed: tracking control error: {e}");
            warn!("Suggestions:");
            warn!("  - Check fgs_server is running and accessible");
            warn!("  - Verify the --fgs-server-url is correct");
            Err(format!("Tracking error: {e}").into())
        }
        Err(StaticCalibrationError::TrackingReacquireFailed) => {
            error!("Calibration failed: tracking did not reacquire within timeout");
            warn!("Suggestions:");
            warn!("  - Increase --lock-on-time-secs");
            warn!("  - Check that the spot is still visible");
            warn!("  - Reduce --step-amplitude-urad if spot moves out of frame");
            Err("Tracking reacquire failed".into())
        }
        Err(StaticCalibrationError::DisplayError(e)) => {
            error!("Calibration failed: display control error: {e}");
            warn!("Suggestions:");
            warn!("  - Check calibrate_serve is running and accessible");
            warn!("  - Verify the --calibrate-serve-url is correct");
            Err(format!("Display error: {e}").into())
        }
    }
}
