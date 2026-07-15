# demo06: OpenCL + OpenGL Interop (Zero-Copy GPU→GPU)

N-body gravity simulation where the OpenCL kernel writes particle positions **directly** into an OpenGL Vertex Buffer Object (VBO). No CPU readback — the GPU computes physics and renders from the same buffer.

## Architecture

```
  ┌─────────────┐         ┌──────────────┐
  │  OpenCL     │         │  OpenGL      │
  │  nbody_step │───────▶│  VBO (pos)   │──▶ render points
  │  kernel     │  shared │  (float4 ×N) │
  │  (GPU)      │  buffer │              │
  └─────────────┘         └──────────────┘
         ▲                       ▲
         │   cl_khr_gl_sharing   │
         └───────shared──────────┘
              CL/GL context
```

### Data flow per frame

1. `gl.finish()` — flush pending GL commands
2. `clEnqueueAcquireGLObjects` — CL takes ownership of VBO
3. `clEnqueueNDRangeKernel("nbody_step")` — kernel updates positions in-place
4. `clEnqueueReleaseGLObjects` — GL reclaims ownership of VBO
5. `clFinish` — ensure CL is done
6. GL renders points from VBO via `glDrawArrays(GL_POINTS)`
7. `swap_buffers`

### Particle data layout

Each particle is a `float4`:
- `.xyz` — position
- `.w` — mass

Velocity is stored in a separate CL-only buffer (not shared with GL).

### N-body kernel

O(N²) gravitational simulation with:
- Plummer softening (`softening_sq`) to avoid singularities
- Velocity damping (`× 0.999` per step)
- Wall bounce at ±`box_half` with configurable `restitution`

## Key implementation details

### CL-GL context sharing

On Linux, the OpenCL context must be created with properties pointing to the **active** GL context and display:

- **GLX**: `CL_GL_CONTEXT_KHR` + `CL_GLX_DISPLAY_KHR`
- **EGL**: `CL_GL_CONTEXT_KHR` + `CL_EGL_DISPLAY_KHR`

The code extracts raw handles via `glutin::context::AsRawContext` and `glutin::display::AsRawDisplay`, then matches at runtime.

### Dual-GPU systems (Intel iGPU + NVIDIA dGPU)

On systems with both Intel and NVIDIA GPUs, the GL context may land on the Intel iGPU by default. Intel's OpenCL driver typically lacks `cl_khr_gl_sharing`, causing `CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR`.

**Fix**: Set PRIME offload environment variables *before* creating the GL context:

```bash
__NV_PRIME_RENDER_OFFLOAD=1
__GLX_VENDOR_LIBRARY_NAME=nvidia
```

The code also iterates through all OpenCL platforms/devices and tries to create a CL-GL interop context with each one, selecting the first that succeeds.

### VAO requirement (OpenGL 3.3 Core)

A Vertex Array Object (VAO) must be bound before calling `glVertexAttribPointer`. Without it, the call is silently ignored in core profile, resulting in a black screen.

### Queue consistency

The same CL queue must be used for `gl_acquire`, kernel enqueue, and `gl_release`. The code uses `ProQue::queue()` for all three operations.

## Dependencies

| Crate       | Version | Purpose                          |
|-------------|---------|----------------------------------|
| `ocl`       | 0.19    | OpenCL bindings (context, buffer, kernel) |
| `glutin`    | 0.32    | OpenGL context creation          |
| `glutin-winit` | 0.5 | glutin + winit integration       |
| `winit`     | 0.30    | Window and event loop            |
| `glow`      | 0.14    | OpenGL function loading and calls |
| `raw-window-handle` | 0.6 | Raw platform handles for interop |

## Running

```bash
cargo run -p demo06_opencl_opengl_interop
```

A window opens with ~2048 blue particles in an n-body simulation, bouncing off invisible walls. Console output shows:
- Which GL backend (GLX/EGL) is used
- Which OpenCL platform/device succeeded
- GL vendor and renderer (confirms GPU selection)
- Per-frame GPU step time (~0.4 ms for 2048 particles on GTX 1650)

## Expected output

```
Raw GL context: Glx(0x...)
Raw GL display: Glx(0x...)
Using GLX interop: ctx=0x..., display=0x...
Found 2 OpenCL platforms
  Platform 0: Intel(R) OpenCL HD Graphics (OpenCL 3.0)
  Platform 1: NVIDIA CUDA (OpenCL 3.0 CUDA 12.1.98)
  Trying platform Intel... -> Failed: CreateContextClGlSharingUnsupported
  Trying platform NVIDIA... -> SUCCESS!
GL vendor: NVIDIA Corporation
GL renderer: NVIDIA GeForce GTX 1650/PCIe/SSE2
GPU step: 0.42 ms (N=2048, zero-copy GL-CL interop)
```

## Problems encountered and lessons learned

This demo went through several rounds of debugging. The issues and their solutions are documented here to save time in future CL-GL interop projects.

### 1. API mismatches across crate versions (winit 0.30, glutin 0.32, glow 0.14)

**Problem**: The `winit`, `glutin`, and `glutin-winit` crates had breaking API changes between versions. Code examples from older tutorials used APIs that no longer exist.

**Symptoms**: Compilation errors for `WindowBuilder` (replaced by `WindowAttributes`), `gl_config.display()` (method not found without correct trait imports), `window.get_proc_address()` (moved to `gl_display.get_proc_address()`).

**Fix**: Carefully read the actual crate source in `~/.cargo/registry/` to discover the correct API. Key discoveries:
- `winit 0.30`: `WindowBuilder` → `WindowAttributes::default()`
- `glutin 0.32`: Need `use glutin::display::GetGlDisplay;` for `.display()`, `use glutin::context::AsRawContext;` for `.raw_context()`
- `glutin-winit 0.5`: `DisplayBuilder::build()` closure parameter `|configs|` must be `|mut configs|` to call `.next()`

**Lesson**: When using rapidly-evolving crates, read the actual source in the cargo registry rather than relying on docs/examples that may be outdated. The crate source is the ground truth.

### 2. `CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR` (-1000) at CL context creation

**Problem**: OpenCL context creation with GL sharing properties failed on one platform but succeeded on another.

**Root cause**: Dual-GPU laptop (Intel iGPU + NVIDIA dGPU). The GL context was created on the Intel iGPU by default, but Intel's OpenCL driver does not support `cl_khr_gl_sharing`. Only the NVIDIA platform supports it.

**Symptoms**: `OclCore(ApiWrapper(CreateContextClGlSharingUnsupported))` on Intel platform. Even after finding the NVIDIA platform, `CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR` persisted because the GL context was still on Intel.

**Fix (two-part)**:
1. **Force GL context onto NVIDIA** via PRIME offload environment variables, set *before* creating the GL context:
   ```rust
   std::env::set_var("__NV_PRIME_RENDER_OFFLOAD", "1");
   std::env::set_var("__GLX_VENDOR_LIBRARY_NAME", "nvidia");
   ```
2. **Iterate all CL platforms** and attempt context creation with each, rather than assuming the first platform is the right one. Include `CL_CONTEXT_PLATFORM` in the properties — it is required for GL sharing to work.

**Lesson**: On dual-GPU systems, the GL context and CL context **must** be on the same physical GPU. Verify with `glGetString(GL_VENDOR)` / `glGetString(GL_RENDERER)` after context creation. If they don't match the CL device, interop will fail silently or with cryptic error codes.

### 3. `CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR` at kernel enqueue (not context creation)

**Problem**: CL context creation succeeded, `from_gl_buffer` succeeded, but `clEnqueueNDRangeKernel` failed with the same -1000 error.

**Root cause**: The `ProQue` creates its own internal CL queue, but we were using a separately created `cl_queue` for `gl_acquire`/`gl_release`. The acquire and kernel enqueue were on **different queues** — the GL objects were acquired on one queue but the kernel ran on another.

**Fix**: Use the ProQue's own queue for all CL operations:
```rust
let cl_queue = pro_que.queue().clone();
```
Then use `cl_queue` for both `gl_acquire`/`gl_release` and ensure the kernel (built from `pro_que`) runs on the same queue.

**Lesson**: In OpenCL, `clEnqueueAcquireGLObjects` and the kernel that uses the acquired buffer **must** be enqueued on the same command queue. If using a high-level wrapper like `ProQue`, make sure your manual queue operations use `ProQue::queue()`, not a separately created queue.

### 4. Black screen (no rendering) despite successful interop

**Problem**: The window opened, CL kernel ran (console showed step times), but the screen was entirely black — no points visible.

**Root cause**: OpenGL 3.3 Core profile **requires** a bound VAO (Vertex Array Object) before calling `glVertexAttribPointer`. Without a VAO, the call is silently ignored — no error is generated, but no vertex data is uploaded to the pipeline. The draw call produces no fragments.

**Fix**: Create and bind a VAO before setting vertex attributes:
```rust
let vao = unsafe { gl.create_vertex_array().unwrap() };
// In render loop:
gl.bind_vertex_array(Some(vao));
gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 0, 0);
```

**Lesson**: OpenGL Core profile silently ignores many operations that would work in compatibility profile. Always create a VAO early in your setup. The lack of a GL error makes this particularly hard to debug — the driver is technically correct, just unhelpful.

### 5. `glow::get_parameter_string` return type mismatch

**Problem**: Attempted to cast the return value of `gl.get_parameter_string()` to `*const i8` and wrap with `CStr::from_ptr`, as one would with raw GL `glGetString`.

**Root cause**: `glow 0.14` returns a `String` directly from `get_parameter_string`, not a raw pointer. The old C API pattern doesn't apply.

**Fix**: Use the `String` directly:
```rust
let gl_vendor = unsafe { gl.get_parameter_string(glow::VENDOR) };
println!("GL vendor: {}", gl_vendor);
```

**Lesson**: High-level GL wrappers like `glow` abstract away the raw C API. Don't assume the C API patterns transfer — check the wrapper's actual return types.

### 6. GLX vs EGL backend selection

**Problem**: Initially tried forcing EGL via `ApiPreference::PreferEgl` to work around the dual-GPU issue. This caused `CL_INVALID_OPERATION` because the EGL display handle didn't match what the NVIDIA OpenCL driver expected for GL sharing.

**Fix**: Revert to default GLX preference (which matches NVIDIA's driver model on Linux X11). The PRIME offload env vars work with GLX, not EGL.

**Lesson**: On Linux with NVIDIA, GLX is the preferred backend for CL-GL interop. EGL interop with NVIDIA OpenCL requires specific EGL implementation support that may not be available. When debugging interop issues, try the default backend first before forcing an alternative.

## General takeaways for CL-GL interop projects

1. **Verify GPU alignment early**: Print `GL_VENDOR`/`GL_RENDERER` and CL platform/device names. If they don't match, no amount of property tweaking will fix interop.

2. **Read crate source in `~/.cargo/registry/`**: When docs are outdated or incomplete, the actual source is the only reliable reference. This was essential for discovering `AsRawContext`, `AsRawDisplay`, `RawContext::Glx`, and `RawDisplay::Glx` enums.

3. **Runtime-match GLX vs EGL**: Don't hardcode one backend. Use `raw_context()` / `raw_display()` and match on the enum variants at runtime. This makes the code portable across different Linux display server configurations.

4. **Always include `CL_CONTEXT_PLATFORM`**: When creating a CL context with GL sharing, the platform property is mandatory. Without it, the implementation may pick the wrong platform and fail with a generic error.

5. **Flush GL before CL acquire**: Call `gl.finish()` before `clEnqueueAcquireGLObjects`. Pending GL commands that write to the shared buffer must complete before CL takes ownership.

6. **Use a single CL queue for all interop operations**: Acquire, kernel, and release must share one queue. If using `ProQue`, derive your queue from `pro_que.queue()`.

7. **Always create a VAO in OpenGL Core**: Even if you only have one buffer and one attribute. Without it, `glVertexAttribPointer` is a no-op.

8. **Iterate platforms, don't assume**: On multi-vendor systems, only one CL platform will support GL sharing with a given GL context. Try all of them and take the first success.

9. **Set PRIME env vars before GL context creation**: `std::env::set_var` must be called before `EventLoop::new()` / `DisplayBuilder::build()`. Setting them after the GL context exists has no effect.

10. **Check `glow` return types**: `glow` wraps the C API with Rust types. Functions like `get_parameter_string` return `String`, not `*const u8`. Don't apply C API patterns to the wrapper.
