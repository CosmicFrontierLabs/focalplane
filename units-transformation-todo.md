# Units Transformation TODO

This document tracks the migration to type-safe units using the `uom` crate to prevent unit confusion errors.

## Completed Migrations

### ✅ Temperature Functions (Priority 1)
- **Completed**: All functions taking `temp_c: f64`
- **Implementation**: Using `ThermodynamicTemperature` type from uom
- **Extension trait**: `TemperatureExt` for convenient Celsius/Kelvin/Fahrenheit conversions
- **Files updated**: 
  - `simulator/src/units.rs` - Core Temperature type and TemperatureExt trait
  - `simulator/src/hardware/dark_current.rs` - Dark current temperature calculations
  - All sensor models and temperature-dependent calculations

### ✅ Pixel Size Units (Priority 2 - Partial)
- **Completed**: `pixel_size_um: f64` → `pixel_size: Length`
- **Implementation**: Using `Length` type from uom with micrometers as common unit
- **Extension trait**: `LengthExt` for nm/μm/mm/cm/m conversions
- **Files updated**:
  - `simulator/src/units.rs` - Core Length type and LengthExt trait  
  - `simulator/src/hardware/sensor.rs` - All sensor models updated
  - All usage sites updated to use `Length::from_micrometers()`
- **Refactored conversions**:
  - Manual `/1000.0` → `as_millimeters()`
  - Manual `/10000.0` → `as_centimeters()`
  - Manual `/1_000_000.0` → `as_meters()`

## Priority 1: Critical Unit Confusion Areas

These are areas where unit mix-ups are most likely and dangerous:

### Magnitude and Color Index Functions
- [ ] `BlackbodyStellarSpectrum::from_gaia_bv_magnitude(bv: f64, mag: f64)`
  - **Risk**: Arguments can be swapped (already happened!)
  - **Solution**: Use `ColorIndex` and `Magnitude` types


### Wavelength Parameters
- [ ] Functions with `wavelength_nm: f64`
  - **Risk**: Could pass meters or angstroms
  - **Solution**: Use `Length` type with nanometer units

### Angular Conversions
- [ ] `to_radians()`, `to_degrees()`, `from_radians()`, `from_degrees()`
  - **Risk**: Double conversion or wrong direction
  - **Solution**: Use `Angle` type throughout

## Priority 2: Physical Dimensions

### Length Units
- [ ] Telescope aperture (`aperture_m: f64`)
- [ ] Focal length (`focal_length_m: f64`)
- [x] ~~Pixel size (`pixel_size_um: f64`)~~ ✅ COMPLETED
- [ ] Sensor dimensions (`width_um`, `height_um`)
- [ ] Wavelength bands (`lower_nm`, `upper_nm`)

### Time Units
- [ ] Exposure duration (`Duration` already type-safe!)
- [ ] Frame rates (`max_frame_rate_fps: f64`)
- [ ] Dark current rates (electrons/second)

### Area Units
- [ ] Collecting area (`collecting_area_cm2()`)
- [ ] Aperture calculations

## Priority 3: Detector Units

### Electronic Units
- [ ] Well depth (`max_well_depth_e: f64`)
- [ ] Dark current (e⁻/pixel/second)
- [ ] Read noise (e⁻ RMS)
- [ ] ADC gain (`dn_per_electron: f64`)

### Flux Rates
- [ ] Photon flux (photons/s/cm²)
- [ ] Electron flux (e⁻/s/cm²)
- [ ] Irradiance (erg/s/cm²/Hz)

## Priority 4: Derived Units

### Optical Parameters
- [ ] F-number (dimensionless, but could be typed)
- [ ] Plate scale (arcsec/mm)
- [ ] Pixel scale (arcsec/pixel)
- [ ] Field of view (degrees)

### Motion Units
- [ ] Angular velocity (rad/s)
- [ ] Wobble frequency (Hz)
- [ ] RPM conversions

## Implementation Strategy

### Phase 1: Core Types
1. Define custom unit types for astronomy:
   ```rust
   type Magnitude = f64; // Will be refined
   type ColorIndex = f64; // Will be refined
   type QuantumEfficiency = f64; // 0.0-1.0
   ```

2. Create astronomy-specific units:
   ```rust
   unit! {
       system: uom::si;
       quantity: uom::si::luminous_intensity;
       @magnitude: 1.0; "mag";
       @jansky: 1.0e-26; "Jy";
   }
   ```

### Phase 2: Critical Functions
1. Update function signatures with highest risk first
2. Add validation at boundaries
3. Update tests to use typed units

### Phase 3: Full Migration
1. Replace all `f64` parameters with appropriate unit types
2. Update documentation
3. Add compile-time unit checking

## Files to Modify (by priority)

### High Priority
- `simulator/src/photometry/stellar.rs` - magnitude/color functions
- `simulator/src/hardware/dark_current.rs` - temperature units
- `simulator/src/photometry/spectrum.rs` - wavelength units
- `simulator/src/star_math.rs` - angular conversions

### Medium Priority
- `simulator/src/hardware/telescope.rs` - aperture, focal length
- `simulator/src/hardware/sensor.rs` - pixel size, dimensions
- `simulator/src/photometry/photoconversion.rs` - flux calculations
- `simulator/src/image_proc/airy.rs` - wavelength, pixel units

### Lower Priority
- `simulator/src/algo/motion.rs` - angular velocity
- `simulator/src/photometry/quantum_efficiency.rs` - wavelengths
- All test files - update to use typed units

## Testing Strategy

1. Create unit conversion tests
2. Verify dimensional analysis at compile time
3. Add boundary condition tests with units
4. Performance benchmarks (ensure no runtime overhead)

## Benefits

- **Compile-time safety**: Can't mix up units
- **Self-documenting**: Types show units clearly
- **Conversion handling**: Automatic unit conversions
- **Scientific accuracy**: Proper dimensional analysis

## Notes

- Start with most error-prone areas first
- Keep performance in mind (zero-cost abstractions)
- Document custom units clearly
- Consider gradual migration with type aliases first