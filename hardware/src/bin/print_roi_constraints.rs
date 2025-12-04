use anyhow::Result;
use clap::Parser;
use hardware::nsv455::camera::nsv455_camera::NSV455Camera;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "/dev/video0")]
    device: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("NSV455 ROI Constraints");
    println!("Device: {}\n", args.device);

    let camera = NSV455Camera::from_device(args.device)?;
    let constraints = camera.roi_constraints();

    println!("Horizontal Offset Constraints:");
    println!("  Minimum:        {} pixels", constraints.h_offset.min);
    println!("  Maximum:        {} pixels", constraints.h_offset.max);
    println!("  Step/Alignment: {} pixels", constraints.h_offset.step);

    println!("\nVertical Offset Constraints:");
    println!("  Minimum:        {} pixels", constraints.v_offset.min);
    println!("  Maximum:        {} pixels", constraints.v_offset.max);
    println!("  Step/Alignment: {} pixels", constraints.v_offset.step);

    println!("\nSupported ROI Sizes:");
    for (width, height) in &constraints.supported_sizes {
        println!("  {width}x{height}");
    }

    Ok(())
}
