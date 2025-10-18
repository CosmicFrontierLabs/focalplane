#[cfg(feature = "hardware-tests")]
mod tests {
    use nsv455::camera::neutralino_imx455::read_sensor_temperatures;

    #[test]
    fn test_read_sensor_temperatures() {
        let device_path = "/dev/video0";

        let (fpga_temp, pcb_temp) = read_sensor_temperatures(device_path);

        println!("FPGA Temperature: {:?}°C", fpga_temp);
        println!("PCB Temperature: {:?}°C", pcb_temp);

        if fpga_temp.is_some() {
            let temp = fpga_temp.unwrap();
            assert!(
                temp > -50.0 && temp < 150.0,
                "FPGA temperature out of reasonable range"
            );
        }

        if pcb_temp.is_some() {
            let temp = pcb_temp.unwrap();
            assert!(
                temp > -50.0 && temp < 150.0,
                "PCB temperature out of reasonable range"
            );
        }
    }
}
