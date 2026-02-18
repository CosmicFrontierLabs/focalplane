//! FSM (Fast Steering Mirror) control types.

use serde::{Deserialize, Serialize};

/// FSM controller status response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FsmStatus {
    /// Whether FSM is connected and initialized
    pub connected: bool,
    /// Current X position in microradians
    pub x_urad: f64,
    /// Current Y position in microradians
    pub y_urad: f64,
    /// Minimum X position in microradians
    pub x_min: f64,
    /// Maximum X position in microradians
    pub x_max: f64,
    /// Minimum Y position in microradians
    pub y_min: f64,
    /// Maximum Y position in microradians
    pub y_max: f64,
    /// Last error message (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

impl Default for FsmStatus {
    fn default() -> Self {
        Self {
            connected: false,
            x_urad: 0.0,
            y_urad: 0.0,
            x_min: 0.0,
            x_max: 2000.0,
            y_min: 0.0,
            y_max: 2000.0,
            last_error: None,
        }
    }
}

/// Request to move FSM to a position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FsmMoveRequest {
    /// Target X position in microradians
    pub x_urad: f64,
    /// Target Y position in microradians
    pub y_urad: f64,
}

/// Request to connect or disconnect the FSM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FsmConnectRequest {
    /// True to connect, false to disconnect
    pub connected: bool,
}
