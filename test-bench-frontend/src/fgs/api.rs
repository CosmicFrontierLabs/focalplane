use gloo_net::http::Request;

/// Fetch text content from a URL, returning None on failure.
pub async fn fetch_text(url: &str) -> Option<String> {
    let response = Request::get(url).send().await.ok()?;
    response.text().await.ok()
}
