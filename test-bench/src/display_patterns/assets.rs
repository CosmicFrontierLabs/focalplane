use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "display_assets/"]
pub struct Assets;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_embedded_assets() {
        println!("\nEmbedded assets:");
        for file in Assets::iter() {
            println!("  - {file}");
        }

        let count = Assets::iter().count();
        println!("\nTotal embedded assets: {count}");
        assert!(count > 0, "No assets were embedded!");
    }
}
