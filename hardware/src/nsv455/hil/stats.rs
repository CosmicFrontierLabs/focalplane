pub struct FrameStats {
    pub total_frames: u64,
    pub last_frame_time: std::time::Instant,
    pub fps_samples: Vec<f32>,
    pub last_histogram: Vec<u32>,
    pub fpga_temp_celsius: Option<f32>,
    pub pcb_temp_celsius: Option<f32>,
    pub last_temp_update: std::time::Instant,
    pub last_timestamp_sec: i64,
    pub last_timestamp_usec: i64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            total_frames: 0,
            last_frame_time: std::time::Instant::now(),
            fps_samples: Vec::with_capacity(10),
            last_histogram: vec![0; 256],
            fpga_temp_celsius: None,
            pcb_temp_celsius: None,
            last_temp_update: std::time::Instant::now(),
            last_timestamp_sec: 0,
            last_timestamp_usec: 0,
        }
    }
}
