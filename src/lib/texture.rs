use image::GenericImageView;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler
}

impl Texture {
    pub fn from_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], label: &str) -> Result<Self, image::ImageError> {
        let img = image::load_from_memory(bytes)?;
        Texture::from_image(device, queue, &img, Some(label))
    }

    pub fn from_raw_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: Vec<u8>, width: u32, height: u32, label: &str) -> Result<Self, image::ImageError> {
        if let Some(img) = image::RgbaImage::from_raw(width, height, bytes) {
            Texture::from_image(device, queue, &image::DynamicImage::ImageRgba8(img), Some(label))
        }
        else {
            Err(image::ImageError::Parameter(image::error::ParameterError::from_kind(image::error::ParameterErrorKind::DimensionMismatch)))
        }
    }

    pub fn from_image(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::DynamicImage, label: Option<&str>) -> Result<Self, image::ImageError> {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1
        };

        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size,
                label,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
            }
        );
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1)
            },
            size
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self { texture, view, sampler })
    }
}
