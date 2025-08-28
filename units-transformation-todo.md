# Units Transformation TODO

This document tracks the migration to type-safe units using the `uom` crate to prevent unit confusion errors.

## Summary (Last Updated: 2025-08-27)

**Major Progress**: The critical physics units are now type-safe! Temperature, Length, Wavelength, Angle, and Area types have been fully implemented with convenient extension traits. The most error-prone areas (telescope optics, sensor geometry, angular calculations) have been successfully migrated.

**Statistics**:
- **5 core unit types** implemented with extension traits
- **111+ unit conversion calls** across 21+ files
- **30+ files** now using typed units
- **10+ files** fully migrated to type-safe units

**Remaining Work**: Primarily electronic sensor parameters (well depth, ADC gain) and domain-specific astronomical units (Magnitude, ColorIndex). The foundation is solid and most physics calculations are now protected from unit confusion errors.

## Completed Migrations

### ✅ Temperature Functions (Priority 1)
- **Completed**: All functions taking `temp_c: f64`
- **Implementation**: Using `ThermodynamicTemperature` type from uom
- **Extension trait**: `TemperatureExt` for convenient Celsius/Kelvin/Fahrenheit conversions
- **Files updated**: 
  - `simulator/src/units.rs` - Core Temperature type and TemperatureExt trait
  - `simulator/src/hardware/dark_current.rs` - Dark current temperature calculations
  - All sensor models and temperature-dependent calculations
- **Usage**: 111+ conversion calls across 21 files

### ✅ Length Units (Priority 2)
- **Completed**: Full `Length` type implementation with `LengthExt` trait
- **Pixel Size**: `pixel_size_um: f64` → `pixel_size: Length`
- **Telescope Optics**: 
  - `aperture_m: f64` → `aperture: Length`
  - `focal_length_m: f64` → `focal_length: Length`
- **Sensor Dimensions**: Using `SensorGeometry` struct with typed `Length` fields
- **Extension trait**: `LengthExt` for nm/μm/mm/cm/m conversions
- **Files updated**:
  - `simulator/src/units.rs` - Core Length type and LengthExt trait  
  - `simulator/src/hardware/sensor.rs` - All sensor models updated
  - `simulator/src/hardware/telescope.rs` - TelescopeConfig fully typed
  - All usage sites updated to use typed constructors
- **Refactored conversions**:
  - Manual `/1000.0` → `as_millimeters()`
  - Manual `/10000.0` → `as_centimeters()`
  - Manual `/1_000_000.0` → `as_meters()`

### ✅ Wavelength Units (Priority 1)
- **Completed**: Type alias `Wavelength = Length` for optical clarity
- **Implementation**: All wavelength parameters use `Wavelength` type
- **Usage**: `wavelength_nm: f64` → `wavelength: Wavelength`
- **Files updated**:
  - All photometry modules (`spectrum.rs`, `stellar.rs`, `photoconversion.rs`)
  - Airy disk calculations in `image_proc/airy.rs`
  - Telescope diffraction calculations
- **Key functions**:
  - `wavelength_to_ergs(wavelength: Wavelength) -> f64`
  - `airy_disk_radius(&self, wavelength: Wavelength) -> Angle`
  - `quantum_efficiency.at(wavelength: Wavelength) -> f64`

### ✅ Angular Units (Priority 1)
- **Completed**: All angular measurements migrated (commit 4a3f770)
- **Implementation**: `Angle` type with `AngleExt` trait
- **Conversions**: degrees ↔ radians ↔ arcseconds ↔ milliarcseconds
- **Files updated**:
  - `simulator/src/star_math.rs` - `field_diameter()`, `pixel_scale()` return `Angle`
  - `simulator/src/hardware/satellite.rs` - FOV and plate scale methods
  - `simulator/src/hardware/telescope.rs` - Airy disk and resolution calculations
  - `simulator/src/algo/motion.rs` - Angular velocities
- **Key functions**:
  - `field_diameter(telescope, sensor) -> Angle`
  - `pixel_scale(telescope, sensor) -> Angle`
  - `plate_scale() -> Angle`
  - `field_of_view() -> (Angle, Angle)`
  - `airy_disk_radius(wavelength) -> Angle`

### ✅ Area Units (Priority 2)
- **Completed**: Area measurements for apertures (commit 9f2bf02)
- **Implementation**: `Area` type with `AreaExt` trait
- **Conversions**: square meters ↔ square centimeters
- **Files updated**:
  - `simulator/src/hardware/telescope.rs` - `clear_aperture_area() -> Area`
  - `simulator/src/photometry/photoconversion.rs` - `integrated_over(aperture: Area)`
- **Key functions**:
  - `clear_aperture_area() -> Area`
  - `photons(spectrum, band, aperture: Area, exposure) -> f64`

## Remaining Work

### Priority 1: Critical Unit Confusion Areas

#### Magnitude and Color Index Functions
- [ ] `BlackbodyStellarSpectrum::from_gaia_bv_magnitude(bv: f64, mag: f64)`
  - **Risk**: Arguments can be swapped (already happened!)
  - **Solution**: Use `ColorIndex` and `Magnitude` types
  - **Status**: Not yet started - still using raw f64

### Priority 2: Physical Dimensions (Partially Complete)

#### Time Units
- [x] ~~Exposure duration (`Duration` already type-safe!)~~ ✅ Rust std::time::Duration
- [ ] Frame rates (`max_frame_rate_fps: f64`)
- [ ] Dark current rates (electrons/second) - partially done, temperature typed

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

## Implementation Progress

### ✅ Phase 1: Core Types (COMPLETED)
- **Temperature**: `ThermodynamicTemperature` with `TemperatureExt` trait
- **Length/Wavelength**: `Length` type with `LengthExt` trait, `Wavelength` alias
- **Angle**: `Angle` type with `AngleExt` trait (degrees, radians, arcseconds, mas)
- **Area**: `Area` type with `AreaExt` trait

### 🔄 Phase 2: Critical Functions (IN PROGRESS)
- ✅ Temperature-dependent calculations all typed
- ✅ Wavelength parameters all typed
- ✅ Angular conversions all typed
- ✅ Telescope optics (aperture, focal length) typed
- ✅ Sensor geometry typed via `SensorGeometry` struct
- ⏳ Magnitude/ColorIndex types pending
- ⏳ Electronic units pending

### Phase 3: Full Migration (PLANNED)
1. Complete electronic sensor parameters
2. Create domain-specific types for magnitudes
3. Migrate remaining f64 parameters

## Files Status

### ✅ Completed Files
- `simulator/src/units.rs` - Core unit system with all extension traits
- `simulator/src/hardware/telescope.rs` - Fully typed (aperture, focal length)
- `simulator/src/hardware/sensor.rs` - SensorGeometry with typed dimensions
- `simulator/src/hardware/dark_current.rs` - Temperature typed
- `simulator/src/star_math.rs` - Angular conversions complete
- `simulator/src/photometry/spectrum.rs` - Wavelength units complete
- `simulator/src/photometry/photoconversion.rs` - Area and wavelength typed
- `simulator/src/image_proc/airy.rs` - Wavelength typed
- `simulator/src/algo/motion.rs` - Angular velocity typed
- `simulator/src/photometry/quantum_efficiency.rs` - Wavelengths typed

### 🔄 Partially Complete
- `simulator/src/photometry/stellar.rs` - Wavelength typed, magnitude/color pending

### ❌ Not Started
- Electronic sensor parameters across various files
- Magnitude and ColorIndex custom types

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