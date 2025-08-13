use clap::Parser;
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::telescope::models::IDEAL_50CM;
use simulator::hardware::SatelliteConfig;
use simulator::units::{LengthExt, Temperature, TemperatureExt, Wavelength};

#[derive(Parser, Debug)]
#[command(name = "Telescope Sampling Info")]
struct Args {
    /// Target FWHM sampling in pixels per FWHM
    #[arg(long, default_value_t = 2.0)]
    fwhm_sampling: f64,

    /// Wavelength in nanometers
    #[arg(long, default_value_t = 550.0)]
    wavelength: f64,
}

fn main() {
    let args = Args::parse();

    println!(
        "Telescope configurations for {:.3} pixels per FWHM sampling:",
        args.fwhm_sampling
    );
    println!("Wavelength: {:.0}nm", args.wavelength);
    println!();
    println!(
        "{:<20} {:>15} {:>15} {:>15} {:>20} {:>15} {:>15}",
        "Sensor",
        "Focal Length (m)",
        "Aperture (m)",
        "f-number",
        "FWHM (pixels)",
        "FOV (sr)",
        "Sky Coverage %"
    );
    println!("{:-<115}", "");

    for sensor in sensor_models::ALL_SENSORS.iter() {
        // Create base satellite config
        let base_satellite = SatelliteConfig::new(
            IDEAL_50CM.clone(),
            sensor.clone(),
            Temperature::from_celsius(0.0), // Temperature doesn't affect optical calculations
            Wavelength::from_nanometers(args.wavelength),
        );

        // Adjust to match desired sampling
        let adjusted_satellite = base_satellite.with_fwhm_sampling(args.fwhm_sampling);

        let focal_length = adjusted_satellite.telescope.focal_length.as_meters();
        let aperture = adjusted_satellite.telescope.aperture.as_meters();
        let f_number = adjusted_satellite.telescope.f_number();
        let fwhm_pixels = adjusted_satellite.airy_disk_fwhm_sampled().fwhm();

        // Use the new field_of_view_steradians method
        let fov_steradians = adjusted_satellite.field_of_view_steradians();

        // Calculate percentage of full sky (4π steradians)
        let sky_coverage_percent = (fov_steradians / (4.0 * std::f64::consts::PI)) * 100.0;

        // Debug: get FOV in arcmin to check
        let (fov_x_arcmin, fov_y_arcmin) = adjusted_satellite.field_of_view_arcmin();

        println!(
            "{:<20} {:>15.8} {:>15.2} {:>15.1} {:>20.3} {:>15.6} {:>15.8} ({:.1}' x {:.1}')",
            sensor.name,
            focal_length,
            aperture,
            f_number,
            fwhm_pixels,
            fov_steradians,
            sky_coverage_percent,
            fov_x_arcmin,
            fov_y_arcmin
        );
    }
}
