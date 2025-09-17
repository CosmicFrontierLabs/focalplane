use crate::v4l2_capture::CaptureSession;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::stats::FrameStats;

pub struct AppState {
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub padding: u32,
    pub session: Arc<Mutex<CaptureSession<'static>>>,
    pub device_path: String,
    pub stats: Arc<Mutex<FrameStats>>,
}
