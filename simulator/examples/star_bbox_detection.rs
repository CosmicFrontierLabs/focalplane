use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;
use simulator::image_proc::thresholding::{apply_threshold, connected_components, otsu_threshold};
use simulator::image_proc::{
    aabbs_to_tuples, convolve2d, detect_stars, draw_bounding_boxes, draw_simple_boxes,
    draw_stars_with_x_markers, gaussian_kernel, get_bounding_boxes, merge_overlapping_aabbs,
    ConvolveMode, ConvolveOptions, StarDetection,
};
use std::collections::HashMap;

fn main() {
    // Create test image with stars
    let input_image = create_test_image();

    // Get output directory and save original image
    let output_dir = test_helpers::get_output_dir();
    let original_path = output_dir.join("original_starfield.png");
    input_image
        .save(&original_path)
        .expect("Failed to save original image");
    println!("Saved original image to: {}", original_path.display());

    // Convert to ndarray for processing
    let gray_image = input_image.to_luma8();
    let height = gray_image.height() as usize;
    let width = gray_image.width() as usize;

    let image_array = Array2::from_shape_fn((height, width), |(y, x)| {
        gray_image.get_pixel(x as u32, y as u32).0[0] as f64 / 255.0
    });

    // Apply Gaussian blur to reduce noise
    let kernel = gaussian_kernel(5, 1.5);
    let options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };

    let smoothed = convolve2d(&image_array.view(), &kernel.view(), Some(options));

    // Threshold the image
    let threshold = otsu_threshold(&smoothed.view());
    println!("Otsu threshold: {:.4}", threshold);

    let binary = apply_threshold(&smoothed.view(), threshold);

    // Connected components labeling
    let labels = connected_components(&binary.view());

    // Get bounding boxes
    let bboxes = get_bounding_boxes(&labels.view());
    println!("Found {} initial bounding boxes", bboxes.len());

    // Also detect star centroids and properties using moment analysis
    let stars = detect_stars(&smoothed.view(), Some(threshold));
    println!("Detected {} stars using moment analysis", stars.len());

    // Merge overlapping boxes with a small padding
    let merged_aabbs = merge_overlapping_aabbs(&bboxes, Some(2));
    println!("After merging: {} bounding boxes", merged_aabbs.len());

    // Convert to tuples for compatibility with drawing functions
    let merged_bboxes = aabbs_to_tuples(&merged_aabbs);

    // Create labels with diameters for the merged boxes
    let mut diameter_labels = Vec::new();
    let mut star_circles = Vec::new();

    for &(min_row, min_col, max_row, max_col) in &merged_bboxes {
        // Find stars within this box
        let box_stars: Vec<&StarDetection> = stars
            .iter()
            .filter(|star| {
                let x = star.x as usize;
                let y = star.y as usize;
                x >= min_col && x <= max_col && y >= min_row && y <= max_row
            })
            .collect();

        // Calculate average diameter of stars in this box
        let diameter = if !box_stars.is_empty() {
            let avg_diameter =
                box_stars.iter().map(|s| s.diameter).sum::<f64>() / box_stars.len() as f64;
            format!("Φ{:.1}", avg_diameter)
        } else {
            "?".to_string()
        };

        diameter_labels.push(diameter);

        // Add all stars in this box to the circles list
        for star in box_stars {
            // Using the exact star centroid coordinates for precise positioning
            star_circles.push((star.y, star.x, star.diameter));

            // Print the position and diameter for debugging
            println!(
                "Star at ({:.1}, {:.1}) with diameter {:.1}",
                star.y, star.x, star.diameter
            );
        }
    }

    // Expand bounding boxes by 2 pixels in all directions
    let expanded_bboxes: Vec<(usize, usize, usize, usize)> = merged_bboxes
        .iter()
        .map(|(min_row, min_col, max_row, max_col)| {
            (
                min_row.saturating_sub(2),
                min_col.saturating_sub(2),
                (*max_row + 2).min(input_image.height() as usize - 1),
                (*max_col + 2).min(input_image.width() as usize - 1),
            )
        })
        .collect();

    // Create four output images:
    // 1. Original boxes (red)
    // 2. Merged boxes with diameter labels (pale green)
    // 3. Merged boxes with actual star size circles (pale blue)
    // 4. Stars with X markers and numbers (orange)

    // Convert original boxes to tuples for drawing
    let original_box_tuples = aabbs_to_tuples(&bboxes);
    let original_boxes_image = draw_simple_boxes(&input_image, &original_box_tuples, (255, 0, 0));

    // Use pale green (144, 238, 144) for boxes
    let merged_boxes_image = draw_bounding_boxes(
        &input_image,
        &expanded_bboxes,
        (144, 238, 144),
        Some(&diameter_labels),
        None,
        None,
    );

    // Use pale blue (135, 206, 250) for circles
    let result_image = draw_bounding_boxes(
        &input_image,
        &expanded_bboxes,
        (144, 238, 144),
        Some(&diameter_labels),
        Some(&star_circles),
        Some((135, 206, 250)),
    );

    // Create labeled stars with diameter information
    let mut labeled_stars = HashMap::new();
    for &(y, x, diameter) in &star_circles {
        // Format diameter to 1 decimal place
        let label = format!("D={:.1}", diameter);
        labeled_stars.insert(label, (y, x, diameter));
    }

    // Use light blue (135, 206, 250) for X markers
    let x_markers_image = draw_stars_with_x_markers(
        &input_image,
        &labeled_stars,
        (135, 206, 250), // Light blue color
        1.0,             // Arm length factor (1.0 = full diameter)
    );

    // Text rendering is now working properly with high quality options

    // Save all four results
    let original_boxes_path = output_dir.join("stars_original_boxes.png");
    original_boxes_image
        .save(&original_boxes_path)
        .expect("Failed to save original boxes image");

    let merged_boxes_path = output_dir.join("stars_merged_boxes.png");
    merged_boxes_image
        .save(&merged_boxes_path)
        .expect("Failed to save merged boxes image");

    let circles_path = output_dir.join("stars_with_circles.png");
    result_image
        .save(&circles_path)
        .expect("Failed to save image with circles");

    let x_markers_path = output_dir.join("stars_with_x_markers.png");
    x_markers_image
        .save(&x_markers_path)
        .expect("Failed to save image with X markers");

    println!("Processing complete. Results saved to:");
    println!(
        "  - Original boxes (red): {}",
        original_boxes_path.display()
    );
    println!(
        "  - Merged boxes (pale green): {}",
        merged_boxes_path.display()
    );
    println!(
        "  - Star sizes (pale blue circles): {}",
        circles_path.display()
    );
    println!(
        "  - Star X markers (light blue): {}",
        x_markers_path.display()
    );
}

// Test function removed - text rendering now working properly

// Create a test image with simulated stars
fn create_test_image() -> DynamicImage {
    let width = 512;
    let height = 512;
    let mut image = GrayImage::new(width, height);

    // Fill with dark background (simulating night sky)
    for pixel in image.pixels_mut() {
        *pixel = Luma([15]);
    }

    // Add stars of various sizes and brightnesses
    add_star(&mut image, 100, 100, 3, 220);
    add_star(&mut image, 150, 250, 2, 200);
    add_star(&mut image, 200, 150, 4, 240);
    add_star(&mut image, 250, 300, 3, 210);
    add_star(&mut image, 300, 200, 5, 250);
    add_star(&mut image, 350, 250, 2, 190);
    add_star(&mut image, 400, 350, 3, 230);
    add_star(&mut image, 150, 400, 4, 220);
    add_star(&mut image, 450, 150, 2, 200);
    add_star(&mut image, 200, 450, 3, 210);

    // Simulate a brighter area (like a galaxy)
    add_galaxy(&mut image, 350, 400, 30, 50);

    // Add some noise (simulating sensor noise)
    add_noise(&mut image, 8);

    DynamicImage::ImageLuma8(image)
}

// Helper function to add a star with Gaussian profile
fn add_star(image: &mut GrayImage, x: u32, y: u32, radius: u32, intensity: u8) {
    let width = image.width();
    let height = image.height();

    for dy in -(radius as i32 * 3)..=(radius as i32 * 3) {
        for dx in -(radius as i32 * 3)..=(radius as i32 * 3) {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                let distance = ((dx * dx + dy * dy) as f32).sqrt();

                // Create a Gaussian profile
                let falloff = (-distance * distance / (radius as f32 * 2.0).powi(2)).exp();
                let value = (intensity as f32 * falloff) as u8;

                if value > 0 {
                    let current = image.get_pixel(px as u32, py as u32).0[0];
                    let new_value = current.saturating_add(value);
                    image.put_pixel(px as u32, py as u32, Luma([new_value]));
                }
            }
        }
    }
}

// Helper function to add a galaxy-like feature
fn add_galaxy(image: &mut GrayImage, x: u32, y: u32, radius: u32, intensity: u8) {
    let width = image.width();
    let height = image.height();

    for dy in -(radius as i32)..=(radius as i32) {
        for dx in -(radius as i32)..=(radius as i32) {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                let distance = ((dx * dx + dy * dy) as f32).sqrt();

                if distance <= radius as f32 {
                    // Create a smoothed disk with some irregular structure
                    let angle = (dx as f32).atan2(dy as f32);
                    let spiral = (distance * 0.2 + angle * 2.0).sin() * 0.2 + 0.8;
                    let falloff = (1.0 - (distance / radius as f32).powi(2)) * spiral;
                    let value = (intensity as f32 * falloff) as u8;

                    let current = image.get_pixel(px as u32, py as u32).0[0];
                    let new_value = current.saturating_add(value);
                    image.put_pixel(px as u32, py as u32, Luma([new_value]));
                }
            }
        }
    }

    // Add a few bright spots within the galaxy
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();

    for _ in 0..15 {
        let r = rng.gen::<f32>() * radius as f32 * 0.8;
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;

        let sx = x as i32 + (r * theta.cos()) as i32;
        let sy = y as i32 + (r * theta.sin()) as i32;

        if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
            add_star(
                image,
                sx as u32,
                sy as u32,
                1,
                intensity + rng.gen_range(0..20),
            );
        }
    }
}

// Add random noise to the image
fn add_noise(image: &mut GrayImage, amplitude: u8) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for pixel in image.pixels_mut() {
        let noise = rng.gen_range(0..amplitude * 2) as i16 - amplitude as i16;
        let current = pixel.0[0] as i16;
        let new_value = (current + noise).clamp(0, 255) as u8;
        *pixel = Luma([new_value]);
    }
}
