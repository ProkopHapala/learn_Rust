//! # OpenCL + OpenGL Interop Demo: Zero-Copy N-Body Simulation
//!
//! N-body gravity simulation where an OpenCL kernel writes particle positions
//! directly into an OpenGL Vertex Buffer Object (VBO). The GPU computes physics
//! and renders from the same buffer — no CPU readback.
//!
//! ## Architecture
//!
//! ```text
//!   OpenCL kernel (nbody_step)
//!        │
//!        ▼  clEnqueueAcquireGLObjects
//!   Shared VBO (float4 × N)  ←── cl_khr_gl_sharing context
//!        │
//!        ▼  clEnqueueReleaseGLObjects
//!   OpenGL render (glDrawArrays GL_POINTS)
//! ```
//!
//! ## Key features
//!
//! - **Zero-copy interop**: CL kernel writes directly into GL buffer via `clCreateFromGLBuffer`
//! - **Dual-GPU support**: PRIME offload env vars force GL context onto NVIDIA dGPU
//! - **Platform auto-selection**: iterates all CL platforms to find one with `cl_khr_gl_sharing`
//! - **GLX/EGL runtime detection**: matches raw context/display handles at runtime
//! - **Wall bounce**: particles bounce off invisible walls at ±`box_half` with energy loss
//!
//! ## Per-frame data flow
//!
//! 1. `glFinish()` — flush pending GL commands
//! 2. `clEnqueueAcquireGLObjects` — CL takes ownership of VBO
//! 3. `clEnqueueNDRangeKernel` — kernel updates positions in-place
//! 4. `clEnqueueReleaseGLObjects` — GL reclaims ownership
//! 5. `clFinish` — ensure CL is done
//! 6. GL renders points from VBO
//! 7. `swap_buffers`
//!
//! ## Particle layout
//!
//! Each particle is a `float4`: `.xyz` = position, `.w` = mass.
//! Velocity is stored in a separate CL-only buffer (not shared with GL).

use rand::Rng;
use std::time::Instant;

use ocl::{ProQue, Buffer, flags::MemFlags};
use ocl::core::ContextProperties;

use glutin::context::{AsRawContext, GlProfile, NotCurrentGlContext};
use glutin::display::{GetGlDisplay, AsRawDisplay};
use glutin::prelude::*;
use glutin::surface::GlSurface;

use glutin_winit::GlWindow;

use raw_window_handle::HasWindowHandle;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowAttributes;

use glow::HasContext;

/// OpenCL C kernel source for the n-body gravity simulation.
///
/// Each work-item computes the net gravitational force on one particle `i`
/// from all other particles `j`, then updates velocity and position via
/// semi-implicit Euler integration.
///
/// # Parameters
/// - `pos` — `float4` array: `.xyz` = position, `.w` = mass. Shared GL VBO.
/// - `vel` — `float4` array: `.xyz` = velocity, `.w` = unused. CL-only buffer.
/// - `n` — number of particles.
/// - `dt` — time step.
/// - `softening_sq` — Plummer softening squared (prevents singularity at r→0).
/// - `box_half` — half-size of the bounding cube; walls at ±`box_half`.
/// - `restitution` — velocity retention on wall bounce (0.0 = stop, 1.0 = elastic).
///
/// # Physics
///
/// Force on particle `i` from `j`:
/// ```text
/// F = G * m_i * m_j * r̂ / (|r|² + ε²)^(3/2)
/// ```
/// where `r = pos[j] - pos[i]`, `ε` = softening. Here G=1 (natural units).
///
/// Integration: semi-implicit Euler (`v += a·dt; x += v·dt`)
/// with velocity damping (`× 0.999`) and wall bounce.
const KERNEL_SRC: &str = r#"
__kernel void nbody_step(
    __global float4* pos,
    __global float4* vel,
    const int n,
    const float dt,
    const float softening_sq,
    const float box_half,
    const float restitution
) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 pi = pos[i];
    float3 fi = (float3)(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float4 pj = pos[j];
        float3 r = pj.xyz - pi.xyz;
        float dist_sq = dot(r, r) + softening_sq;
        float inv_dist = rsqrt(dist_sq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float f = pi.w * pj.w * inv_dist3;
        fi += r * f;
    }
    float4 v = vel[i];
    v.xyz += fi * dt / pi.w;
    v.xyz *= 0.999f;
    float4 p = pi;
    p.xyz += v.xyz * dt;
    // Bounce off box walls at ±box_half
    if (p.x < -box_half) { p.x = -box_half; if (v.x < 0.0f) v.x = -v.x * restitution; }
    if (p.x >  box_half) { p.x =  box_half; if (v.x > 0.0f) v.x = -v.x * restitution; }
    if (p.y < -box_half) { p.y = -box_half; if (v.y < 0.0f) v.y = -v.y * restitution; }
    if (p.y >  box_half) { p.y =  box_half; if (v.y > 0.0f) v.y = -v.y * restitution; }
    if (p.z < -box_half) { p.z = -box_half; if (v.z < 0.0f) v.z = -v.z * restitution; }
    if (p.z >  box_half) { p.z =  box_half; if (v.z > 0.0f) v.z = -v.z * restitution; }
    pos[i] = p;
    vel[i] = v;
}
"#;

/// Number of particles in the simulation.
const N: usize = 2048;

/// Entry point: sets up GL window, CL interop context, shared VBO, and render loop.
///
/// # Pipeline overview
///
/// 1. **PRIME offload** — set env vars to force NVIDIA GPU on dual-GPU systems
/// 2. **GL window** — create via `glutin-winit` with OpenGL Core profile
/// 3. **CL-GL interop context** — extract raw GLX/EGL handles, iterate CL platforms
/// 4. **Shared VBO** — GL creates buffer, CL wraps it via `clCreateFromGLBuffer`
/// 5. **Shaders** — simple vertex/fragment for point rendering
/// 6. **Main loop** — acquire→kernel→release→render→swap each frame
fn main() {
    // Force NVIDIA GPU for GL context on dual-GPU systems (PRIME offload).
    // Without this, GL context lands on Intel iGPU which lacks cl_khr_gl_sharing,
    // causing CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR at context creation.
    std::env::set_var("__NV_PRIME_RENDER_OFFLOAD", "1");
    std::env::set_var("__GLX_VENDOR_LIBRARY_NAME", "nvidia");

    let event_loop = EventLoop::new().unwrap();

    // === 1. Create GL window via glutin-winit ===
    let (window, gl_config) = {
        let window_attrs = WindowAttributes::default()
            .with_title("OpenCL+OpenGL Interop N-Body")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let display_builder = glutin_winit::DisplayBuilder::new()
            .with_window_attributes(Some(window_attrs));

        let (window, gl_config) = display_builder
            .build(&event_loop, <_>::default(), |mut configs| {
                configs.next().unwrap()
            })
            .unwrap();

        (window.unwrap(), gl_config)
    };

    let gl_display = gl_config.display(); // needs GetGlDisplay trait from glutin::prelude::*

    // Create GL context
    let gl_context = {
        let attrs = glutin::context::ContextAttributesBuilder::new()
            .with_profile(GlProfile::Core)
            .build(Some(window.window_handle().unwrap().as_raw()));
        unsafe { gl_display.create_context(&gl_config, &attrs).unwrap() }
    };

    // Create GL surface
    let surface_attrs = window.build_surface_attributes(<_>::default()).unwrap();
    let gl_surface = unsafe { gl_display.create_window_surface(&gl_config, &surface_attrs).unwrap() };

    // Make context current
    let gl_context = gl_context.make_current(&gl_surface).unwrap();

    // === 2. Extract raw GL context + display handles for CL-GL interop ===
    // On Linux, OpenCL needs the raw GLX context (GLXContext) and display (Display*)
    // or EGL context and display, passed as CL context properties.
    let raw_ctx = gl_context.raw_context();
    let raw_gl_display = gl_display.raw_display();

    println!("Raw GL context: {:?}", raw_ctx);
    println!("Raw GL display: {:?}", raw_gl_display);

    // Build CL context properties based on whether we're using GLX or EGL.
    // The match is runtime — glutin selects the backend based on platform/driver availability.
    let mut props = ContextProperties::new();
    match (raw_ctx, raw_gl_display) {
        (glutin::context::RawContext::Glx(ctx_ptr), glutin::display::RawDisplay::Glx(disp_ptr)) => {
            println!("Using GLX interop: ctx={:p}, display={:p}", ctx_ptr, disp_ptr);
            props.set_gl_context(ctx_ptr as *mut std::ffi::c_void);
            props.set_glx_display(disp_ptr as *mut std::ffi::c_void);
        }
        (glutin::context::RawContext::Egl(ctx_ptr), glutin::display::RawDisplay::Egl(disp_ptr)) => {
            println!("Using EGL interop: ctx={:p}, display={:p}", ctx_ptr, disp_ptr);
            props.set_gl_context(ctx_ptr as *mut std::ffi::c_void);
            props.set_egl_display(disp_ptr as *mut std::ffi::c_void);
        }
        _ => panic!("Unsupported GL backend combination: ctx={:?}, display={:?}", raw_ctx, raw_gl_display),
    }

    let platforms = ocl::Platform::list();
    println!("Found {} OpenCL platforms", platforms.len());
    for (idx, p) in platforms.iter().enumerate() {
        println!("  Platform {}: {:?} (version: {:?})", idx, p.name(), p.version());
    }

    // Try each CL platform/device — only the one matching the GL device will support
    // cl_khr_gl_sharing. We iterate all platforms and attempt context creation with
    // CL_CONTEXT_PLATFORM set, until one succeeds.
    let mut cl_context = None;
    let mut platform = None;
    let mut device = None;
    for p in &platforms {
        let devs = ocl::Device::list_all(p).unwrap_or_default();
        for d in devs {
            println!("  Trying platform {:?} device {:?}...", p.name(), d.name());
            // Must include CL_CONTEXT_PLATFORM in properties for GL sharing to work
            let mut p2 = props.clone();
            p2.set_platform(*p);
            let result = ocl::Context::builder()
                .platform(*p)
                .properties(p2)
                .build();
            match result {
                Ok(ctx) => {
                    println!("  -> SUCCESS! CL-GL interop context created.");
                    cl_context = Some(ctx);
                    platform = Some(*p);
                    device = Some(d);
                    break;
                }
                Err(e) => {
                    println!("  -> Failed: {:?}", e);
                }
            }
        }
        if cl_context.is_some() { break; }
    }

    let cl_context = cl_context.expect("No OpenCL platform supports CL-GL sharing with this GL context");
    let _platform = platform.unwrap();
    let _device = device.unwrap();

    let pro_que = ProQue::builder()
        .context(cl_context.clone())
        .src(KERNEL_SRC)
        .dims(N)
        .build().unwrap();

    // Use the ProQue's own queue for all CL operations (kernel + GL acquire/release)
    let cl_queue = pro_que.queue().clone();

    // === 4. Create GL context loader and VBO for particle positions ===
    // `glow` loads GL function pointers via the glutin display's get_proc_address.
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            gl_display.get_proc_address(std::ffi::CString::new(s).unwrap().as_c_str()).cast()
        })
    };

    // Verify which GPU the GL context is on
    let gl_vendor = unsafe { gl.get_parameter_string(glow::VENDOR) };
    let gl_renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };
    println!("GL vendor: {}", gl_vendor);
    println!("GL renderer: {}", gl_renderer);

    let mut rng = rand::thread_rng();
    let pos_host: Vec<f32> = (0..N).flat_map(|_| {
        let x = rng.gen_range(-0.8..0.8);
        let y = rng.gen_range(-0.8..0.8);
        let z = rng.gen_range(-0.3..0.3);
        let m = rng.gen_range(0.5..2.0);
        [x, y, z, m]
    }).collect();
    let vel_host: Vec<f32> = (0..N).flat_map(|_| {
        [rng.gen_range(-0.05..0.05), rng.gen_range(-0.05..0.05), 0.0f32, 0.0]
    }).collect();

    // Create GL VBO
    let vbo = unsafe { gl.create_buffer().unwrap() };
    unsafe {
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let pos_bytes: &[u8] = std::slice::from_raw_parts(
            pos_host.as_ptr() as *const u8,
            pos_host.len() * std::mem::size_of::<f32>(),
        );
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, pos_bytes, glow::DYNAMIC_COPY);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    // === 5. Create CL buffer from GL VBO (the interop step!) ===
    // `clCreateFromGLBuffer` wraps the GL buffer as a CL memory object.
    // The CL buffer shares the same GPU memory — no copy.
    // Must call glFinish() first to ensure GL commands are flushed.
    unsafe { gl.finish(); }
    let gl_vbo_id: u32 = vbo.0.get();
    println!("Creating CL buffer from GL VBO id={}...", gl_vbo_id);
    let pos_cl_buf = Buffer::<f32>::from_gl_buffer(
        &cl_queue,
        Some(MemFlags::new().read_write()),
        gl_vbo_id,
    ).expect("Failed to create CL buffer from GL VBO");
    println!("CL buffer from GL VBO created successfully! len={}", pos_cl_buf.len());

    // CL-only velocity buffer (not shared with GL — no need to render velocities).
    let vel_cl_buf = unsafe {
        Buffer::<f32>::new(
            &cl_queue,
            MemFlags::new().read_write().copy_host_ptr(),
            N * 4,
            Some(&vel_host),
        ).unwrap()
    };

    let kernel = pro_que.kernel_builder("nbody_step")
        .arg(&pos_cl_buf)
        .arg(&vel_cl_buf)
        .arg(N as i32)
        .arg(0.005f32)
        .arg(0.05f32 * 0.05f32)
        .arg(1.0f32)   // box_half
        .arg(0.8f32)   // restitution
        .build().unwrap();

    // === 6. GL render setup: shaders + VAO ===
    // Vertex shader: maps pos_mass.xyz to clip space, sets point size.
    // Fragment shader: outputs a fixed blue color for all points.
    let vertex_shader = unsafe {
        let shader = gl.create_shader(glow::VERTEX_SHADER).unwrap();
        gl.shader_source(shader, "#version 330 core\nlayout(location=0) in vec4 pos_mass;\nvoid main() {\n    gl_Position = vec4(pos_mass.xyz, 1.0);\n    gl_PointSize = 4.0;\n}");
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            panic!("Vertex shader compile error: {}", gl.get_shader_info_log(shader));
        }
        shader
    };

    let fragment_shader = unsafe {
        let shader = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
        gl.shader_source(shader, "#version 330 core\nout vec4 frag_color;\nvoid main() { frag_color = vec4(0.4, 0.7, 1.0, 1.0); }");
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            panic!("Fragment shader compile error: {}", gl.get_shader_info_log(shader));
        }
        shader
    };

    let program = unsafe {
        let prog = gl.create_program().unwrap();
        gl.attach_shader(prog, vertex_shader);
        gl.attach_shader(prog, fragment_shader);
        gl.link_program(prog);
        if !gl.get_program_link_status(prog) {
            panic!("Program link error: {}", gl.get_program_info_log(prog));
        }
        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);
        prog
    };

    // VAO (Vertex Array Object) — required in OpenGL 3.3+ Core profile.
    // Without a bound VAO, glVertexAttribPointer is silently ignored → black screen.
    let vao = unsafe { gl.create_vertex_array().unwrap() };

    let dt = 0.005f32;
    let softening_sq = 0.05f32 * 0.05f32;

    // === 7. Main render loop ===
    // Each frame: CL acquires VBO → kernel updates positions → CL releases VBO → GL renders.
    // The same CL queue must be used for acquire, kernel, and release (ProQue's queue).
    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                target.exit();
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // --- CL step: acquire GL buffer, run kernel, release ---
                let t0 = Instant::now();
                unsafe { gl.finish(); } // ensure GL ops complete before CL acquires
                pos_cl_buf.cmd().gl_acquire().enq().expect("gl_acquire failed");
                kernel.set_arg(3, dt).unwrap();
                kernel.set_arg(4, softening_sq).unwrap();
                kernel.set_arg(5, 1.0f32).unwrap();       // box_half
                kernel.set_arg(6, 0.8f32).unwrap();       // restitution
                unsafe { kernel.enq().expect("kernel enqueue failed"); }
                pos_cl_buf.cmd().gl_release().enq().expect("gl_release failed");
                cl_queue.finish().expect("queue finish failed");
                let ms = t0.elapsed().as_secs_f32() * 1000.0;

                // --- GL render: draw from the shared VBO (no CPU readback!) ---
                unsafe {
                    gl.clear_color(0.05, 0.05, 0.08, 1.0);
                    gl.clear(glow::COLOR_BUFFER_BIT);
                    gl.use_program(Some(program));
                    gl.enable(glow::PROGRAM_POINT_SIZE);
                    gl.bind_vertex_array(Some(vao));
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                    gl.enable_vertex_attrib_array(0);
                    gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 0, 0);
                    gl.draw_arrays(glow::POINTS, 0, N as i32);
                    gl.disable_vertex_attrib_array(0);
                    gl.bind_buffer(glow::ARRAY_BUFFER, None);
                    gl.bind_vertex_array(None);
                    gl.use_program(None);
                }

                gl_surface.swap_buffers(&gl_context).ok();

                println!("GPU step: {:.2} ms (N={}, zero-copy GL-CL interop)", ms, N);
            }
            _ => {}
        }
    }).unwrap();
}
