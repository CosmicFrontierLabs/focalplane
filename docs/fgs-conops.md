# Fine Guidance System (FGS) Concept of Operations

## Overview

The Fine Guidance System performs precision pointing control by tracking guide stars and commanding a fine steering mirror (FSM) to maintain telescope alignment.

## Operational Phases

### Phase 1: Initial Acquisition and Calibration

```mermaid
flowchart TD
    Start([FGS Start]) --> Acquire[Acquire N frames]
    Acquire --> Average[Average frames for<br/>noise reduction]
    Average --> Detect[Detect all stars<br/>in averaged frame]
    Detect --> Select[Select guide stars<br/>based on criteria]
    Select --> DefineROI[Define ROIs around<br/>guide stars]
    DefineROI --> Store[Store reference<br/>positions]
    Store --> TrackMode([Enter Tracking Mode])

    style Start fill:#9ca3af,stroke:#6b7280,color:#000
    style TrackMode fill:#86efac,stroke:#65a30d,color:#000
```

#### Guide Star Selection Criteria
- Brightness: Sufficient SNR but not saturated
- Position: Well distributed across field
- Shape: Round PSF (low aspect ratio)
- Isolation: No nearby contaminating sources

### Phase 2: Continuous Tracking Loop

```mermaid
flowchart LR
    Frame([New Frame]) --> ROI[Extract ROIs]
    ROI --> Centroid[Compute centroids<br/>in each ROI]
    Centroid --> Delta[Calculate position<br/>deltas from reference]
    Delta --> Transform[Transform to FSM<br/>coordinates]
    Transform --> Command[Command FSM]
    Command --> Wait[Wait for<br/>next frame]
    Wait --> Frame

    style Frame fill:#fbbf24,stroke:#f59e0b,color:#000
    style Command fill:#fb923c,stroke:#ea580c,color:#000
```

### Phase 3: Detailed Tracking Pipeline

```mermaid
graph TD
    subgraph "Image Acquisition"
        Sensor[Sensor Array] --> Raw[Raw Frame]
        Raw --> Preprocess[Dark/Flat<br/>Correction]
    end

    subgraph "ROI Processing"
        Preprocess --> ROI1[ROI 1]
        Preprocess --> ROI2[ROI 2]
        Preprocess --> ROI3[ROI N]
        
        ROI1 --> C1[Centroid 1]
        ROI2 --> C2[Centroid 2]
        ROI3 --> C3[Centroid N]
    end

    subgraph "Error Calculation"
        C1 --> Err1[Δx₁, Δy₁]
        C2 --> Err2[Δx₂, Δy₂]
        C3 --> Err3[Δxₙ, Δyₙ]
        
        Err1 --> Combine[Weighted<br/>Average]
        Err2 --> Combine
        Err3 --> Combine
        
        Combine --> Global[Global Δx, Δy]
    end

    subgraph "Control Output"
        Global --> PID[PID Controller]
        PID --> FSM[FSM Commands]
        FSM --> Mirror[Fine Steering<br/>Mirror]
    end

    style Sensor fill:#93c5fd,stroke:#3b82f6,color:#000
    style Mirror fill:#c084fc,stroke:#9333ea,color:#000
```

## State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    
    Idle: Idle
    Idle: Waiting for FGS start command
    
    Acquiring: Acquiring  
    Acquiring: Collecting frames for averaging
    
    Calibrating: Calibrating
    Calibrating: - Detecting stars
    Calibrating: - Selecting guides
    Calibrating: - Setting references
    
    Tracking: Tracking
    Tracking: - Continuous centroiding
    Tracking: - FSM commanding
    
    Reacquiring: Reacquiring
    Reacquiring: Attempting to
    Reacquiring: recover lost stars
    
    Idle --> Acquiring: START_FGS
    
    Acquiring --> Calibrating: N_FRAMES_COMPLETE
    Acquiring --> Idle: ABORT
    
    Calibrating --> Tracking: GUIDE_STARS_SELECTED
    Calibrating --> Idle: NO_SUITABLE_STARS
    
    Tracking --> Tracking: PROCESS_FRAME
    Tracking --> Reacquiring: LOST_LOCK
    Tracking --> Idle: STOP_FGS
    
    Reacquiring --> Tracking: LOCK_RECOVERED
    Reacquiring --> Calibrating: TIMEOUT
    Reacquiring --> Idle: ABORT
```

## Data Flow

```mermaid
flowchart TB
    subgraph "Initialization Phase"
        F1[Frame 1] --> Avg[Frame<br/>Averager]
        F2[Frame 2] --> Avg
        FN[Frame N] --> Avg
        Avg --> AvgImg[Averaged<br/>Image]
        AvgImg --> StarDet[Star<br/>Detection]
        StarDet --> GuideList[Guide Star<br/>List]
        GuideList --> ROIDef[ROI<br/>Definitions]
    end

    subgraph "Tracking Phase"
        NewFrame[New Frame] --> ROIExtract[ROI<br/>Extraction]
        ROIDef -.-> ROIExtract
        ROIExtract --> Centroids[Centroid<br/>Calculation]
        RefPos[Reference<br/>Positions] --> ErrorCalc[Error<br/>Calculation]
        Centroids --> ErrorCalc
        ErrorCalc --> FSMCmd[FSM<br/>Command]
    end

    style F1 fill:#86efac,stroke:#65a30d,color:#000
    style F2 fill:#86efac,stroke:#65a30d,color:#000
    style FN fill:#86efac,stroke:#65a30d,color:#000
    style NewFrame fill:#fde047,stroke:#facc15,color:#000
    style FSMCmd fill:#fca5a5,stroke:#ef4444,color:#000
```

## Performance Requirements

### Timing
- Frame rate: 10-100 Hz typical
- Centroid computation: < 1 ms per ROI
- Total loop latency: < 10 ms
- FSM response time: < 5 ms

### Accuracy
- Centroid precision: < 0.05 pixels
- Pointing stability: < 0.1 arcsec RMS
- Guide star minimum SNR: > 20
- Tracking range: ± 10 arcsec

## ROI Management

```mermaid
graph LR
    subgraph "Full Frame"
        FF[4096 x 2300<br/>HWK4123]
    end
    
    subgraph "Guide ROIs"
        ROI1[64 x 64<br/>Star 1]
        ROI2[64 x 64<br/>Star 2]
        ROI3[64 x 64<br/>Star 3]
    end
    
    FF --> ROI1
    FF --> ROI2
    FF --> ROI3
    
    ROI1 --> C1[x₁, y₁]
    ROI2 --> C2[x₂, y₂]
    ROI3 --> C3[x₃, y₃]
    
    C1 --> Proc[Process]
    C2 --> Proc
    C3 --> Proc
```

## Error Recovery

### Lost Guide Star Handling
1. Continue with remaining guide stars if N-1 available
2. Attempt reacquisition in expanded ROI
3. Fall back to rate gyro data if all stars lost
4. Trigger full reacquisition after timeout

### Performance Degradation
- 3+ guide stars: Nominal performance
- 2 guide stars: Reduced accuracy, no rotation control
- 1 guide star: Position hold only
- 0 guide stars: Open loop with gyro data

## Implementation Considerations

### Guide Star Selection Algorithm
```
1. Detect all stars in averaged frame
2. Filter by brightness (mag_min < m < mag_max)
3. Filter by shape (aspect_ratio < 1.5)
4. Sort by brightness
5. Select up to N_max stars with good spatial distribution
6. Define ROI as 3σ × PSF_FWHM around each star
```

### Centroid Algorithm Options
- Intensity-weighted center of mass (current implementation)
- Gaussian PSF fitting
- Quadratic interpolation
- Correlation with reference PSF

### FSM Command Generation
```
Δx_fsm = K_p × Δx_avg + K_i × ∫Δx dt + K_d × dΔx/dt
Δy_fsm = K_p × Δy_avg + K_i × ∫Δy dt + K_d × dΔy/dt
```

Where Δx_avg, Δy_avg are weighted averages across all guide stars.