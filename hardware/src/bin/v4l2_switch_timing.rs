//! V4L2 format switch timing diagnostic
//!
//! Measures per-ioctl timing during full-frame → ROI → full-frame format switch
//! to determine where the ~7.5s stream restart delay originates on the Tegra
//! V4L2 driver with the NSV455/IMX455 sensor.
//!
//! Tests two scenarios:
//! 1. Full teardown/rebuild: Drop MmapStream, set_format, new MmapStream (current approach)
//! 2. Same-format restart: STREAMOFF → STREAMON without format change
//!
//! The goal is to determine whether the delay comes from:
//! - S_FMT ioctl triggering sensor re-initialization
//! - STREAMON ioctl itself
//! - First DQBUF (waiting for sensor to produce a frame)

use anyhow::Result;
use clap::Parser;
use std::time::Instant;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::v4l2::vidioc;
use v4l::video::Capture;

const BUF_COUNT: u32 = 4;

#[derive(Parser, Debug)]
#[command(about = "V4L2 format switch timing diagnostic")]
struct Args {
    /// Video device path
    #[arg(short, long, default_value = "/dev/video0")]
    device: String,

    /// ROI width
    #[arg(long, default_value_t = 128)]
    roi_width: u32,

    /// ROI height
    #[arg(long, default_value_t = 128)]
    roi_height: u32,

    /// Full-frame width
    #[arg(long, default_value_t = 9568)]
    full_width: u32,

    /// Full-frame height
    #[arg(long, default_value_t = 6380)]
    full_height: u32,

    /// Number of frames to capture per phase
    #[arg(short = 'n', long, default_value_t = 3)]
    frames: usize,

    /// Number of full switch cycles to run
    #[arg(long, default_value_t = 3)]
    cycles: usize,
}

struct PhaseTiming {
    label: String,
    set_format_ms: f64,
    mmap_stream_new_ms: f64,
    first_frame_ms: f64,
    subsequent_frame_ms: Vec<f64>,
    stream_drop_ms: f64,
}

impl PhaseTiming {
    fn total_to_first_frame(&self) -> f64 {
        self.set_format_ms + self.mmap_stream_new_ms + self.first_frame_ms
    }
}

fn print_phase(timing: &PhaseTiming) {
    println!("  {} format switch breakdown:", timing.label);
    println!("    set_format:       {:>8.1}ms", timing.set_format_ms);
    println!("    MmapStream::new:  {:>8.1}ms", timing.mmap_stream_new_ms);
    println!("    first frame:      {:>8.1}ms", timing.first_frame_ms);
    for (i, ms) in timing.subsequent_frame_ms.iter().enumerate() {
        println!("    frame {}:          {:>8.1}ms", i + 2, ms);
    }
    println!("    stream drop:      {:>8.1}ms", timing.stream_drop_ms);
    println!(
        "    TOTAL to 1st frm: {:>8.1}ms",
        timing.total_to_first_frame()
    );
    println!();
}

/// Set format and capture frames, measuring each step independently
fn timed_capture(
    device: &Device,
    width: u32,
    height: u32,
    num_frames: usize,
    label: &str,
) -> Result<PhaseTiming> {
    // Step 1: set_format
    let t0 = Instant::now();
    let mut format = device.format()?;
    format.width = width;
    format.height = height;
    format.fourcc = v4l::FourCC::new(b"Y16 ");
    device.set_format(&format)?;
    let set_format_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Verify
    let actual = device.format()?;
    println!(
        "    [{}] Format: requested {}x{}, got {}x{}, stride={}",
        label, width, height, actual.width, actual.height, actual.stride
    );

    // Step 2: MmapStream::new (REQBUFS + QUERYBUF + mmap)
    let t1 = Instant::now();
    let mut stream = MmapStream::new(device, Type::VideoCapture)?;
    let mmap_stream_new_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Step 3: First frame (includes QBUF + STREAMON + wait for sensor)
    let t2 = Instant::now();
    let (buf, _meta) = stream.next()?;
    let first_frame_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let first_buf_len = buf.len();

    // Step 4: Subsequent frames
    let mut subsequent_frame_ms = Vec::with_capacity(num_frames.saturating_sub(1));
    for _ in 1..num_frames {
        let tf = Instant::now();
        let _ = stream.next()?;
        subsequent_frame_ms.push(tf.elapsed().as_secs_f64() * 1000.0);
    }

    // Step 5: Stream drop (STREAMOFF + munmap + REQBUFS(0))
    let t3 = Instant::now();
    drop(stream);
    let stream_drop_ms = t3.elapsed().as_secs_f64() * 1000.0;

    println!(
        "    [{}] Buffer size: {} bytes ({:.1} MB)",
        label,
        first_buf_len,
        first_buf_len as f64 / 1_048_576.0
    );

    Ok(PhaseTiming {
        label: label.to_string(),
        set_format_ms,
        mmap_stream_new_ms,
        first_frame_ms,
        subsequent_frame_ms,
        stream_drop_ms,
    })
}

/// Test same-format restart (STREAMOFF via drop, then recreate at same resolution)
fn timed_same_format_restart(
    device: &Device,
    _width: u32,
    _height: u32,
    num_frames: usize,
) -> Result<PhaseTiming> {
    // Format is already set from previous capture - just create new stream
    let t1 = Instant::now();
    let mut stream = MmapStream::new(device, Type::VideoCapture)?;
    let mmap_stream_new_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let (buf, _meta) = stream.next()?;
    let first_frame_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let first_buf_len = buf.len();

    let mut subsequent_frame_ms = Vec::with_capacity(num_frames.saturating_sub(1));
    for _ in 1..num_frames {
        let tf = Instant::now();
        let _ = stream.next()?;
        subsequent_frame_ms.push(tf.elapsed().as_secs_f64() * 1000.0);
    }

    let t3 = Instant::now();
    drop(stream);
    let stream_drop_ms = t3.elapsed().as_secs_f64() * 1000.0;

    println!(
        "    [same-fmt restart] Buffer size: {} bytes ({:.1} MB)",
        first_buf_len,
        first_buf_len as f64 / 1_048_576.0
    );

    Ok(PhaseTiming {
        label: "same-fmt restart".to_string(),
        set_format_ms: 0.0,
        mmap_stream_new_ms,
        first_frame_ms,
        subsequent_frame_ms,
        stream_drop_ms,
    })
}

/// Raw ioctl test splitting STREAMON from poll/DQBUF.
/// Uses v4l2-sys-mit types directly for correct struct layout across architectures.
fn timed_raw_capture(device: &Device, width: u32, height: u32, label: &str) -> Result<()> {
    use v4l2_sys_mit::{v4l2_buffer, v4l2_requestbuffers};

    let fd = device.handle().fd();
    let buf_type: u32 = 1; // V4L2_BUF_TYPE_VIDEO_CAPTURE
    let memory: u32 = 1; // V4L2_MEMORY_MMAP

    println!("\n  [{label}] Raw ioctl timing ({width}x{height}):");

    // S_FMT
    let t = Instant::now();
    let mut format = device.format()?;
    format.width = width;
    format.height = height;
    format.fourcc = v4l::FourCC::new(b"Y16 ");
    device.set_format(&format)?;
    println!(
        "    S_FMT:       {:>8.1}ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    // REQBUFS
    let t = Instant::now();
    let mut req: v4l2_requestbuffers = unsafe { std::mem::zeroed() };
    req.count = BUF_COUNT;
    req.type_ = buf_type;
    req.memory = memory;
    let ret = unsafe {
        libc::ioctl(
            fd,
            vidioc::VIDIOC_REQBUFS as libc::c_ulong,
            &mut req as *mut _,
        )
    };
    if ret < 0 {
        anyhow::bail!("REQBUFS failed: {}", std::io::Error::last_os_error());
    }
    println!(
        "    REQBUFS({}): {:>8.1}ms (got {} bufs)",
        BUF_COUNT,
        t.elapsed().as_secs_f64() * 1000.0,
        req.count
    );

    // QUERYBUF + mmap
    let t = Instant::now();
    let mut buf_lengths: Vec<u32> = Vec::new();
    let mut mmap_ptrs: Vec<*mut libc::c_void> = Vec::new();

    for i in 0..req.count {
        let mut v4l2_buf: v4l2_buffer = unsafe { std::mem::zeroed() };
        v4l2_buf.index = i;
        v4l2_buf.type_ = buf_type;
        v4l2_buf.memory = memory;

        let ret = unsafe {
            libc::ioctl(
                fd,
                vidioc::VIDIOC_QUERYBUF as libc::c_ulong,
                &mut v4l2_buf as *mut _,
            )
        };
        if ret < 0 {
            anyhow::bail!("QUERYBUF[{i}] failed: {}", std::io::Error::last_os_error());
        }

        let length = v4l2_buf.length;
        let offset = unsafe { v4l2_buf.m.offset };
        buf_lengths.push(length);

        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                offset as libc::off_t,
            )
        };
        if mmap_ptr == libc::MAP_FAILED {
            anyhow::bail!("mmap[{i}] failed: {}", std::io::Error::last_os_error());
        }
        mmap_ptrs.push(mmap_ptr);
    }
    println!(
        "    QUERYBUF+mmap:{:>8.1}ms ({} bufs, {:.1} MB each)",
        t.elapsed().as_secs_f64() * 1000.0,
        req.count,
        buf_lengths[0] as f64 / 1_048_576.0
    );

    // QBUF all buffers
    let t = Instant::now();
    for i in 0..req.count {
        let mut v4l2_buf: v4l2_buffer = unsafe { std::mem::zeroed() };
        v4l2_buf.index = i;
        v4l2_buf.type_ = buf_type;
        v4l2_buf.memory = memory;

        let ret = unsafe {
            libc::ioctl(
                fd,
                vidioc::VIDIOC_QBUF as libc::c_ulong,
                &mut v4l2_buf as *mut _,
            )
        };
        if ret < 0 {
            anyhow::bail!("QBUF[{i}] failed: {}", std::io::Error::last_os_error());
        }
    }
    println!(
        "    QBUF(all {}): {:>8.1}ms",
        req.count,
        t.elapsed().as_secs_f64() * 1000.0
    );

    // STREAMON
    let t = Instant::now();
    let ret = unsafe {
        libc::ioctl(
            fd,
            vidioc::VIDIOC_STREAMON as libc::c_ulong,
            &buf_type as *const _,
        )
    };
    let streamon_ms = t.elapsed().as_secs_f64() * 1000.0;
    if ret < 0 {
        anyhow::bail!("STREAMON failed: {}", std::io::Error::last_os_error());
    }
    println!("    STREAMON:    {streamon_ms:>8.1}ms");

    // poll() - wait for first buffer
    let t = Instant::now();
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ret = unsafe { libc::poll(&mut pollfd, 1, 15000) };
    let poll_ms = t.elapsed().as_secs_f64() * 1000.0;
    if ret <= 0 {
        anyhow::bail!("poll() failed or timed out: ret={ret}");
    }
    println!("    poll():      {poll_ms:>8.1}ms");

    // DQBUF
    let t = Instant::now();
    let mut v4l2_buf: v4l2_buffer = unsafe { std::mem::zeroed() };
    v4l2_buf.type_ = buf_type;
    v4l2_buf.memory = memory;
    let ret = unsafe {
        libc::ioctl(
            fd,
            vidioc::VIDIOC_DQBUF as libc::c_ulong,
            &mut v4l2_buf as *mut _,
        )
    };
    let dqbuf_ms = t.elapsed().as_secs_f64() * 1000.0;
    if ret < 0 {
        anyhow::bail!("DQBUF failed: {}", std::io::Error::last_os_error());
    }
    println!("    DQBUF:       {dqbuf_ms:>8.1}ms");

    println!(
        "    ------- total STREAMON→frame: {:>8.1}ms",
        streamon_ms + poll_ms + dqbuf_ms
    );

    // Cleanup
    let t = Instant::now();
    unsafe {
        libc::ioctl(
            fd,
            vidioc::VIDIOC_STREAMOFF as libc::c_ulong,
            &buf_type as *const _,
        )
    };
    println!(
        "    STREAMOFF:   {:>8.1}ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    for (i, ptr) in mmap_ptrs.iter().enumerate() {
        unsafe { libc::munmap(*ptr, buf_lengths[i] as usize) };
    }
    req.count = 0;
    unsafe {
        libc::ioctl(
            fd,
            vidioc::VIDIOC_REQBUFS as libc::c_ulong,
            &mut req as *mut _,
        )
    };

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("V4L2 Format Switch Timing Diagnostic");
    println!("Device: {}", args.device);
    println!(
        "Full-frame: {}x{}, ROI: {}x{}",
        args.full_width, args.full_height, args.roi_width, args.roi_height
    );
    println!("Frames per phase: {}, Cycles: {}", args.frames, args.cycles);
    println!("{}", "=".repeat(70));

    let device = Device::with_path(&args.device)?;

    // Warm up with an initial full-frame capture
    println!("\n--- Warm-up (initial full-frame) ---");
    let warmup = timed_capture(
        &device,
        args.full_width,
        args.full_height,
        args.frames,
        "warmup-full",
    )?;
    print_phase(&warmup);

    for cycle in 0..args.cycles {
        println!("=== Cycle {}/{} ===", cycle + 1, args.cycles);

        // Full-frame → ROI switch
        println!("\n--- Full → ROI ---");
        let to_roi = timed_capture(
            &device,
            args.roi_width,
            args.roi_height,
            args.frames,
            "full→ROI",
        )?;
        print_phase(&to_roi);

        // Same-format restart at ROI (no set_format, just new MmapStream)
        println!("--- ROI same-format restart ---");
        let roi_restart =
            timed_same_format_restart(&device, args.roi_width, args.roi_height, args.frames)?;
        print_phase(&roi_restart);

        // ROI → Full-frame switch
        println!("--- ROI → Full ---");
        let to_full = timed_capture(
            &device,
            args.full_width,
            args.full_height,
            args.frames,
            "ROI→full",
        )?;
        print_phase(&to_full);

        // Same-format restart at full-frame
        println!("--- Full same-format restart ---");
        let full_restart =
            timed_same_format_restart(&device, args.full_width, args.full_height, args.frames)?;
        print_phase(&full_restart);

        // Summary for this cycle
        println!("--- Cycle {} Summary ---", cycle + 1);
        println!(
            "  Full→ROI  total-to-first-frame: {:>8.1}ms",
            to_roi.total_to_first_frame()
        );
        println!(
            "  ROI restart (no fmt change):     {:>8.1}ms",
            roi_restart.mmap_stream_new_ms + roi_restart.first_frame_ms
        );
        println!(
            "  ROI→Full  total-to-first-frame: {:>8.1}ms",
            to_full.total_to_first_frame()
        );
        println!(
            "  Full restart (no fmt change):    {:>8.1}ms",
            full_restart.mmap_stream_new_ms + full_restart.first_frame_ms
        );
        println!();
    }

    // Raw ioctl breakdown: STREAMON vs poll vs DQBUF
    println!("{}", "=".repeat(70));
    println!("RAW IOCTL BREAKDOWN (STREAMON vs poll vs DQBUF)");
    println!("{}", "=".repeat(70));

    timed_raw_capture(&device, args.roi_width, args.roi_height, "ROI")?;
    timed_raw_capture(&device, args.full_width, args.full_height, "full-frame")?;
    timed_raw_capture(&device, args.roi_width, args.roi_height, "ROI again")?;

    Ok(())
}
