//! Shared HTTP client infrastructure for both native and WASM targets.
//!
//! This module provides platform-agnostic HTTP helpers that work in both
//! native Rust (using reqwest) and WASM (using gloo-net) environments.

use serde::{de::DeserializeOwned, Serialize};

/// Generic error type for HTTP client operations.
#[derive(Debug, thiserror::Error)]
pub enum HttpClientError {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    Http(String),
    /// Failed to parse response
    #[error("Parse error: {0}")]
    Parse(String),
    /// Connection failed
    #[error("Connection error: {0}")]
    Connection(String),
    /// Request timed out
    #[error("Timeout")]
    Timeout,
    /// Server returned an error status
    #[error("Server error (status {status}): {message}")]
    ServerError { status: u16, message: String },
}

// Platform-specific implementations
#[cfg(target_arch = "wasm32")]
mod platform {
    use super::*;

    impl From<gloo_net::Error> for HttpClientError {
        fn from(err: gloo_net::Error) -> Self {
            HttpClientError::Http(err.to_string())
        }
    }

    #[derive(Debug, Clone)]
    pub struct HttpClient {
        base_url: String,
    }

    impl HttpClient {
        pub fn new(base_url: &str) -> Self {
            Self {
                base_url: base_url.trim_end_matches('/').to_string(),
            }
        }

        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, HttpClientError> {
            use gloo_net::http::Request;

            let url = format!("{}{}", self.base_url, path);
            let response = Request::get(&url).send().await?;

            if !response.ok() {
                return Err(HttpClientError::ServerError {
                    status: response.status(),
                    message: response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string()),
                });
            }

            response
                .json::<T>()
                .await
                .map_err(|e| HttpClientError::Parse(e.to_string()))
        }

        pub async fn post<T: Serialize, R: DeserializeOwned>(
            &self,
            path: &str,
            body: &T,
        ) -> Result<R, HttpClientError> {
            use gloo_net::http::Request;

            let url = format!("{}{}", self.base_url, path);
            let response = Request::post(&url)
                .json(body)
                .map_err(|e| HttpClientError::Parse(e.to_string()))?
                .send()
                .await?;

            if !response.ok() {
                return Err(HttpClientError::ServerError {
                    status: response.status(),
                    message: response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string()),
                });
            }

            response
                .json::<R>()
                .await
                .map_err(|e| HttpClientError::Parse(e.to_string()))
        }

        pub async fn post_no_response<T: Serialize>(
            &self,
            path: &str,
            body: &T,
        ) -> Result<(), HttpClientError> {
            use gloo_net::http::Request;

            let url = format!("{}{}", self.base_url, path);
            let response = Request::post(&url)
                .json(body)
                .map_err(|e| HttpClientError::Parse(e.to_string()))?
                .send()
                .await?;

            if !response.ok() {
                return Err(HttpClientError::ServerError {
                    status: response.status(),
                    message: response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string()),
                });
            }

            Ok(())
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod platform {
    use super::*;

    impl From<reqwest::Error> for HttpClientError {
        fn from(err: reqwest::Error) -> Self {
            if err.is_timeout() {
                HttpClientError::Timeout
            } else if err.is_connect() {
                HttpClientError::Connection(err.to_string())
            } else {
                HttpClientError::Http(err.to_string())
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct HttpClient {
        base_url: String,
        client: reqwest::Client,
    }

    impl HttpClient {
        pub fn new(base_url: &str) -> Self {
            Self {
                base_url: base_url.trim_end_matches('/').to_string(),
                client: reqwest::Client::new(),
            }
        }

        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, HttpClientError> {
            let url = format!("{}{}", self.base_url, path);
            let response = self.client.get(&url).send().await?;

            let status = response.status();
            if !status.is_success() {
                let message = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(HttpClientError::ServerError {
                    status: status.as_u16(),
                    message,
                });
            }

            response
                .json::<T>()
                .await
                .map_err(|e| HttpClientError::Parse(e.to_string()))
        }

        pub async fn post<T: Serialize, R: DeserializeOwned>(
            &self,
            path: &str,
            body: &T,
        ) -> Result<R, HttpClientError> {
            let url = format!("{}{}", self.base_url, path);
            let response = self.client.post(&url).json(body).send().await?;

            let status = response.status();
            if !status.is_success() {
                let message = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(HttpClientError::ServerError {
                    status: status.as_u16(),
                    message,
                });
            }

            response
                .json::<R>()
                .await
                .map_err(|e| HttpClientError::Parse(e.to_string()))
        }

        pub async fn post_no_response<T: Serialize>(
            &self,
            path: &str,
            body: &T,
        ) -> Result<(), HttpClientError> {
            let url = format!("{}{}", self.base_url, path);
            let response = self.client.post(&url).json(body).send().await?;

            let status = response.status();
            if !status.is_success() {
                let message = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(HttpClientError::ServerError {
                    status: status.as_u16(),
                    message,
                });
            }

            Ok(())
        }
    }
}

// Re-export the platform-specific HttpClient
pub use platform::HttpClient;
