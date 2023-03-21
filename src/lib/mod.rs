use std::{num::NonZeroU32, f32::consts::PI};

use rand::{random, thread_rng, Rng};
use wgpu::{include_wgsl, util::DeviceExt};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod texture;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    agent_texture: texture::Texture,
    blur_texture: texture::Texture,

    compute_update_agents: wgpu::ComputePipeline,
    compute_sense_agents: wgpu::ComputePipeline,
    compute_draw_agents: wgpu::ComputePipeline,
    compute_dim_texture: wgpu::ComputePipeline,
    compute_blur_texture: wgpu::ComputePipeline,
    texture_rw_bind_group: wgpu::BindGroup,
    blur_texture_rw_bind_group: wgpu::BindGroup,
    agents_rw_bind_group: wgpu::BindGroup,
    agents_buffer: wgpu::Buffer
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2]
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Agent {
    position: [f32; 2],
    direction: f32
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0,  1.0, 0.0], tex_coords: [0.0, 0.0], }, // A
    Vertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 1.0], }, // B
    Vertex { position: [ 1.0, -1.0, 0.0], tex_coords: [1.0, 1.0], }, // C
    Vertex { position: [ 1.0,  1.0, 0.0], tex_coords: [1.0, 0.0], }, // D
];

const INDICES: &[u16] = &[
    0, 1, 2,
    2, 3, 0
];

const NUM_AGENTS: u32 = 256;
const TEXTURE_SIZE: u32 = 256;

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let agents: Vec<_> = (0..NUM_AGENTS).map(|_| {
            let angle = thread_rng().gen_range(-PI..PI);
            let x = angle.cos() * 40.0;// thread_rng().gen_range(0.0..50.0);
            let y = angle.sin() * 20.0;// thread_rng().gen_range(0.0..50.0);

            Agent { 
                position: [ TEXTURE_SIZE as f32 / 2.0 + x, TEXTURE_SIZE as f32 / 2.0 + y],
                direction: angle - PI/2.0
            }
        }).collect();

        let agents_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Agent Buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            contents: bytemuck::cast_slice(&agents)
        });

        let texture_data: Vec<_> = (0..TEXTURE_SIZE*TEXTURE_SIZE).flat_map(|_| [0, 0, 0, 255]).collect();
        let agent_texture = texture::Texture::from_raw_bytes(&device, &queue, texture_data.clone(), TEXTURE_SIZE, TEXTURE_SIZE, "Some Texture").unwrap();
        let blur_texture = texture::Texture::from_raw_bytes(&device, &queue, texture_data, TEXTURE_SIZE, TEXTURE_SIZE, "Some Texture").unwrap();

        let agent_texture_view = agent_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blur_texture_view = blur_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    // This should match the filterable field of the
                    // corresponding Texture entry above.
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
            label: Some("Texture Bind Group")
        });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&agent_texture.view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&agent_texture.sampler)
                }
            ],
            label: Some("Diffuse Bind Group")
        });

        let texture_rw_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::ReadWrite, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::ReadWrite, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None
                }
            ]
        });

        let agents_rw_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false
                    },
                    count: None
                }
            ]
        });

        let descriptor = wgpu::BindGroupDescriptor {
            layout: &texture_rw_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&agent_texture_view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_texture_view)
                }
            ],
            label: Some("Compute Texture View")
        };

        let texture_rw_bind_group = device.create_bind_group(&descriptor);

        // swap descriptor entries
        let blur_texture_rw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_texture_view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&agent_texture_view)
                }
            ],
            ..descriptor
        });

        let agents_rw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &agents_rw_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agents_buffer.as_entire_binding()
                }
            ],
            label: Some("Compute Texture View")
        });

        let shader = device.create_shader_module(include_wgsl!("./shaders/shader.wgsl"));
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[]
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[ Vertex::desc() ]
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
                })]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            multiview: None
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Descriptor"),
            bind_group_layouts: &[&texture_rw_bind_layout, &agents_rw_bind_layout],
            push_constant_ranges: &[]
        });

        let compute_update_agents = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Agents"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "update_agents"
        });

        let compute_sense_agents = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { 
            label: Some("Sense agents"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "sense_agents"
        });

        let compute_draw_agents = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Draw Agents"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "draw_agents"
        });

        let compute_dim_texture = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dim Texture"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "dim_texture"
        });

        let compute_blur_texture = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dim Texture"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "blur_texture"
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX
            }
        );
        let num_indices = INDICES.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            agent_texture,
            blur_texture,
            compute_update_agents,
            compute_sense_agents,
            compute_draw_agents,
            compute_dim_texture,
            compute_blur_texture,
            texture_rw_bind_group,
            blur_texture_rw_bind_group,
            agents_rw_bind_group,
            agents_buffer
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

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Agents update pass")
        });

        pass.set_bind_group(0, &self.texture_rw_bind_group, &[]);
        pass.set_bind_group(1, &self.agents_rw_bind_group, &[]);

        pass.set_pipeline(&self.compute_update_agents);
        pass.dispatch_workgroups(NUM_AGENTS, 1, 1);

        pass.set_pipeline(&self.compute_sense_agents);
        pass.dispatch_workgroups(NUM_AGENTS, 1, 1);

        pass.set_pipeline(&self.compute_draw_agents);
        pass.dispatch_workgroups(NUM_AGENTS, 1, 1);

        pass.set_pipeline(&self.compute_dim_texture);
        pass.dispatch_workgroups(TEXTURE_SIZE, TEXTURE_SIZE, 1);

        pass.set_pipeline(&self.compute_blur_texture);
        pass.dispatch_workgroups(TEXTURE_SIZE, TEXTURE_SIZE, 1);

        drop(pass);

        self.queue.submit(std::iter::once(encoder.finish()));

        std::mem::swap(&mut self.texture_rw_bind_group, &mut self.blur_texture_rw_bind_group);

    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });


        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.7, g: 0.2, b: 0.4, a: 1.0 }),
                    store: true
                }
            })],
            depth_stencil_attachment: None
        });

        // let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Buffer"),
        //     size: 
        // })
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);


        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_inner_size(winit::dpi::LogicalSize::new(1024, 1024)).build(&event_loop).unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(
        move |event, _, control_flow| match event {
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.update();
                match state.render() {
                    Ok(_) => {},
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        eprintln!("Out of memory!!");
                        *control_flow = ControlFlow::Exit;
                    },
                    Err(e) => eprintln!("{:?}", e)
                }
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    },
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    },
                    _ => {}
                }
            },
            _ => {}
        }
    );
}
