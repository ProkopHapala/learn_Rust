use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};
use std::rc::Rc;
use std::cell::RefCell;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
}

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    particle_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    n_particles: usize,
}

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> Self {
        let size = window.inner_size();
        
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let memory_hints = wgpu::MemoryHints::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Create particles
        let n_particles = 10000;
        let particles: Vec<Particle> = (0..n_particles)
            .map(|_| Particle {
                pos: [rand::random::<f32>() * 2.0 - 1.0, rand::random::<f32>() * 2.0 - 1.0],
                vel: [rand::random::<f32>() * 0.01 - 0.005, rand::random::<f32>() * 0.01 - 0.005],
            })
            .collect();

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Compute shader
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                @group(0) @binding(0)
                var<storage, read_write> particles: array<vec2<f32>>;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i >= arrayLength(&particles) / 2) { return; }
                    
                    let dt = 0.01;
                    let pos_idx = i * 2;
                    let vel_idx = i * 2 + 1;
                    
                    particles[pos_idx] += particles[vel_idx] * dt;
                    
                    // Boundary bounce
                    if (particles[pos_idx].x < -1.0 || particles[pos_idx].x > 1.0) {
                        particles[vel_idx].x *= -0.8;
                        particles[pos_idx].x = clamp(particles[pos_idx].x, -1.0, 1.0);
                    }
                    if (particles[pos_idx].y < -1.0 || particles[pos_idx].y > 1.0) {
                        particles[vel_idx].y *= -0.8;
                        particles[pos_idx].y = clamp(particles[pos_idx].y, -1.0, 1.0);
                    }
                }
            "#)),
        });

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            }],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Render shader
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                struct VertexInput {
                    @location(0) pos: vec2<f32>,
                }

                struct VertexOutput {
                    @builtin(position) clip_pos: vec4<f32>,
                }

                @vertex
                fn vs_main(in: VertexInput) -> VertexOutput {
                    var out: VertexOutput;
                    out.clip_pos = vec4<f32>(in.pos, 0.0, 1.0);
                    return out;
                }

                @fragment
                fn fs_main() -> @location(0) vec4<f32> {
                    return vec4<f32>(0.5, 0.8, 1.0, 1.0);
                }
            "#)),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            particle_buffer,
            compute_pipeline,
            compute_bind_group,
            render_pipeline,
            n_particles,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups((self.n_particles / 64) as u32 + 1, 1, 1);
        }

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
            render_pass.draw(0..self.n_particles as u32, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::Window::new(&event_loop).unwrap();
    window.set_title("WGPU Compute + Render Demo");
    let window_id = window.id();

    let state = Rc::new(RefCell::new(State::new(&window).await));

    event_loop.run(move |event, _target| match event {
        Event::WindowEvent {
            ref event,
            window_id: id,
        } if id == window_id => {
            match event {
                WindowEvent::CloseRequested => _target.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.borrow_mut().resize(*physical_size);
                }
                WindowEvent::RedrawRequested => {
                    state.borrow_mut().render().unwrap();
                }
                _ => {}
            }
        }
        _ => {}
    }).unwrap();
}

fn main() {
    pollster::block_on(run());
}
