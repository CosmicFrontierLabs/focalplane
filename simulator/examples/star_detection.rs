//! Simple star detection example demonstrating centroid-based analysis.

use ndarray::Array2;
use simulator::image_proc::detection::detect_stars;

fn main() {
    println!("Star detection example");

    // Create a synthetic test image (10x10) with a few synthetic stars
    let mut image = Array2::<f64>::zeros((10, 10));

    // Add star 1 - bright point source
    image[[3, 3]] = 0.9;
    image[[2, 3]] = 0.4;
    image[[3, 2]] = 0.4;
    image[[4, 3]] = 0.4;
    image[[3, 4]] = 0.4;

    // Add star 2 - dimmer, slightly elongated
    image[[7, 7]] = 0.6;
    image[[6, 7]] = 0.3;
    image[[7, 6]] = 0.3;
    image[[8, 7]] = 0.3;
    image[[7, 8]] = 0.3;
    image[[8, 8]] = 0.2; // This makes it slightly non-circular

    // Add some noise
    image[[1, 8]] = 0.15;
    image[[8, 1]] = 0.15;

    // Print the image
    println!("Synthetic image:");
    for i in 0..10 {
        for j in 0..10 {
            print!("{:.1} ", image[[i, j]]);
        }
        println!();
    }

    // Detect stars using our algorithm
    let detected_stars = detect_stars(&image.view(), None);

    // Print results
    println!("\nDetected stars:");
    for (i, star) in detected_stars.iter().enumerate() {
        println!(
            "Star {}: position=({:.2}, {:.2}), flux={:.2}, aspect_ratio={:.2}, valid={}",
            i + 1,
            star.x,
            star.y,
            star.flux,
            star.aspect_ratio,
            star.is_valid()
        );

        // Print window around each star
        let x_min = star.x.floor() as usize - 1;
        let x_max = star.x.ceil() as usize + 1;
        let y_min = star.y.floor() as usize - 1;
        let y_max = star.y.ceil() as usize + 1;

        let x_min = x_min.max(0);
        let y_min = y_min.max(0);
        let x_max = x_max.min(image.ncols() - 1);
        let y_max = y_max.min(image.nrows() - 1);

        println!("Window around star {}:", i + 1);
        for i in y_min..=y_max {
            for j in x_min..=x_max {
                print!("{:.1} ", image[[i, j]]);
            }
            println!();
        }
        println!();
    }
}
