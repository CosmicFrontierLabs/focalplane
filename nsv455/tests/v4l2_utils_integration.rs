#[cfg(feature = "hardware-tests")]
mod tests {
    use nsv455::camera::v4l2_utils::{
        collect_camera_metadata, get_available_resolutions, query_menu_item, CameraMetadata,
        Resolution,
    };
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    const DEFAULT_DEVICE: &str = "/dev/video0";

    #[test]
    fn test_get_available_resolutions() {
        let result = get_available_resolutions(DEFAULT_DEVICE);

        match result {
            Ok(resolutions) => {
                println!("Found {} resolutions", resolutions.len());
                for res in &resolutions {
                    match res.fps {
                        Some(fps) => {
                            println!("  {}x{} @ {:.1} fps", res.width, res.height, fps);
                        }
                        None => {
                            println!("  {}x{} @ unknown fps", res.width, res.height);
                        }
                    }
                }

                assert!(
                    !resolutions.is_empty(),
                    "Should find at least one resolution"
                );

                for res in resolutions {
                    assert!(res.width > 0, "Width should be positive");
                    assert!(res.height > 0, "Height should be positive");
                    if let Some(fps) = res.fps {
                        assert!(fps > 0.0, "FPS should be positive if present");
                        assert!(fps < 1000.0, "FPS should be reasonable");
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Could not access device: {}", e);
                eprintln!("This test requires a V4L2 device at /dev/video0");
            }
        }
    }

    #[test]
    fn test_collect_camera_metadata() {
        let result = collect_camera_metadata(DEFAULT_DEVICE);

        match result {
            Ok(metadata) => {
                println!("Camera Metadata:");
                println!("  Driver: {}", metadata.driver);
                println!("  Card: {}", metadata.card);
                println!("  Bus: {}", metadata.bus);
                println!("  Formats: {} found", metadata.formats.len());
                println!("  Controls: {} found", metadata.controls.len());
                println!("  Test Patterns: {} found", metadata.test_patterns.len());

                assert!(
                    !metadata.driver.is_empty(),
                    "Driver name should not be empty"
                );
                assert!(!metadata.card.is_empty(), "Card name should not be empty");
                assert!(!metadata.bus.is_empty(), "Bus info should not be empty");
            }
            Err(e) => {
                eprintln!("Warning: Could not collect metadata: {}", e);
                eprintln!("This test requires a V4L2 device at /dev/video0");
            }
        }
    }

    #[test]
    fn test_query_menu_item_invalid() {
        if let Ok(file) = File::open(DEFAULT_DEVICE) {
            let fd = file.as_raw_fd();

            let result = query_menu_item(fd, 999999, 0);
            assert!(
                result.is_none(),
                "Query with invalid control ID should return None"
            );

            let result = query_menu_item(fd, 0, 999999);
            assert!(
                result.is_none(),
                "Query with invalid index should return None"
            );
        } else {
            eprintln!("Warning: Could not open device for menu item test");
            eprintln!("This test requires a V4L2 device at /dev/video0");
        }
    }

    #[test]
    fn test_resolution_struct() {
        let res_with_fps = Resolution {
            width: 1920,
            height: 1080,
            fps: Some(30.0),
        };

        assert_eq!(res_with_fps.width, 1920);
        assert_eq!(res_with_fps.height, 1080);
        assert_eq!(res_with_fps.fps, Some(30.0));

        let res_without_fps = Resolution {
            width: 4096,
            height: 2160,
            fps: None,
        };

        assert_eq!(res_without_fps.width, 4096);
        assert_eq!(res_without_fps.height, 2160);
        assert_eq!(res_without_fps.fps, None);

        let cloned = res_with_fps.clone();
        assert_eq!(cloned.width, res_with_fps.width);
        assert_eq!(cloned.height, res_with_fps.height);
        assert_eq!(cloned.fps, res_with_fps.fps);
    }

    #[test]
    fn test_camera_metadata_default() {
        let metadata = CameraMetadata::default();

        assert!(metadata.driver.is_empty());
        assert!(metadata.card.is_empty());
        assert!(metadata.bus.is_empty());
        assert!(metadata.formats.is_empty());
        assert!(metadata.resolutions.is_empty());
        assert!(metadata.controls.is_empty());
        assert!(metadata.test_patterns.is_empty());
    }

    #[test]
    fn test_nonexistent_device() {
        let result = get_available_resolutions("/dev/video999");
        assert!(result.is_err(), "Should fail with non-existent device");

        let result = collect_camera_metadata("/dev/video999");
        assert!(result.is_err(), "Should fail with non-existent device");
    }
}
