use clap::Parser;
use hardware::playerone_sdk::Camera;
use tracing::{info, Level};

#[derive(Parser, Debug)]
#[command(name = "playerone_info")]
#[command(about = "Enumerate and display PlayerOne camera properties", long_about = None)]
struct Args {
    /// Show detailed properties for each camera
    #[arg(short, long)]
    detailed: bool,
}

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let args = Args::parse();

    info!("=== PlayerOne Camera Enumeration ===");

    let cameras = Camera::all_cameras();
    let count = cameras.len();

    info!("Found {} PlayerOne camera(s)", count);

    if count == 0 {
        info!("No PlayerOne cameras detected");
        info!("Make sure:");
        info!("  - Camera is connected via USB");
        info!("  - PlayerOne SDK libraries are installed");
        info!("  - You have proper USB permissions");
        return;
    }

    for (i, camera_desc) in cameras.iter().enumerate() {
        info!("");
        info!("=== Camera {} ===", i);

        let props = camera_desc.properties();

        info!("  Camera ID: {}", camera_desc.camera_id());
        info!("  Camera Name: {}", props.camera_model_name);
        info!("  Max Height: {}", props.max_height);
        info!("  Max Width: {}", props.max_width);
        info!("  Pixel Size: {:.2} μm", props.pixel_size);
        info!("  Bit Depth: {}", props.bit_depth);
        info!("  Color Camera: {}", props.is_color_camera);
        info!("  USB3 Speed: {}", props.is_usb_3_speed);
        info!("  Has ST4 Port: {}", props.is_has_st_4_port);

        if args.detailed {
            info!("  Bayer Pattern: {:?}", props.bayer_pattern);
            info!("  Supported Bins: {:?}", props.bins);
            info!("  Image Formats: {:?}", props.img_formats);
        }
    }

    info!("");
    info!("=== Summary ===");
    info!("Total cameras detected: {}", count);
}
