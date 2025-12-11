use anyhow::{Context, Result};
use gpiod::{Chip, Lines, Options, Output};
use std::thread;
use std::time::Duration;

pub struct GpioController {
    chip: Chip,
    line_offset: u32,
    request: Option<Lines<Output>>,
}

impl GpioController {
    pub fn new(chip_name: &str, line_offset: u32) -> Result<Self> {
        let chip = Chip::new(chip_name)
            .with_context(|| format!("Failed to open GPIO chip '{chip_name}'"))?;

        Ok(Self {
            chip,
            line_offset,
            request: None,
        })
    }

    pub fn request_output(&mut self, consumer: &str, initial_value: u8) -> Result<()> {
        let options = Options::output([self.line_offset])
            .values([initial_value != 0])
            .consumer(consumer);

        let request = self
            .chip
            .request_lines(options)
            .with_context(|| "Failed to request GPIO line as output")?;

        self.request = Some(request);
        Ok(())
    }

    pub fn set_value(&mut self, value: u8) -> Result<()> {
        if let Some(ref mut request) = self.request {
            request
                .set_values([value != 0])
                .with_context(|| format!("Failed to set GPIO value to {value}"))?;
        } else {
            anyhow::bail!("GPIO line not requested as output");
        }
        Ok(())
    }

    pub fn blinky(&mut self, on_us: u64, off_us: u64, n: usize) -> Result<()> {
        for i in 0..n {
            self.set_value(1)?;
            thread::sleep(Duration::from_micros(on_us));

            self.set_value(0)?;
            let off_duration = (1 << i) * off_us;
            thread::sleep(Duration::from_micros(off_duration));
        }
        Ok(())
    }
}

pub const ORIN_GPIO_CHIP: &str = "gpiochip0";
pub const ORIN_PX04_LINE: u32 = 127;
