use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "display_assets/"]
pub struct Assets;
