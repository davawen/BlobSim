// Vertex shader

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) tex_coords: vec2<f32>
}

struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
	@location(0) tex_coords: vec2<f32>
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	out.tex_coords = model.tex_coords;
	out.clip_position = vec4<f32>(model.position, 1.0);
	return out;
}

@group(0) @binding(0)
var tex: texture_2d<f32>;
@group(0) @binding(1)
var tex_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	//return vec4<f32>(1.0, 0.0, 0.0, 1.0);
	//textureStore(s_diffuse, vec2<i32>(20, 20), vec4<f32>(1.0, 0.0, 0.0, 1.0));
	return textureSample(tex, tex_sampler, in.tex_coords);
}

fn hash(v: u32) -> u32 {
	var seed: u32 = v;
	seed = (seed ^ u32(61)) ^ (seed >> u32(16));
    seed *= u32(9);
    seed = seed ^ (seed >> u32(4));
    seed *= u32(0x27d4eb2d);
    seed = seed ^ (seed >> u32(15));
    return seed;
}

struct Agent {
	x: f32,
	y: f32,
	direction: f32
}

//@group(0) @binding(0)
//var textureInput: texture_2d<f32>;
@group(0) @binding(0)
var agent_texture: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(0)
var<storage, read_write> agents: array<Agent>;

@compute @workgroup_size(1)
fn update_agents(@builtin(global_invocation_id) id: vec3<u32>) {
	let PI: f32 = 3.14159265359;

	let new_x = agents[id.x].x + cos(agents[id.x].direction);
	let new_y = agents[id.x].y + sin(agents[id.x].direction);

	let dims = textureDimensions(agent_texture);

	if new_x >= f32(dims.x) || new_x < 0.0 || new_y >= f32(dims.y) || new_y < 0.0 { 
		agents[id.x].direction = f32(hash(u32(agents[id.x].direction * 1000.0) + id.x)) / 4294967295.0 * 2.0 * PI;
	}
	else {
		agents[id.x].x = new_x;
		agents[id.x].y = new_y;
	}
}

// @group(0) @binding(0)
// var agent_texture: texture_storage_2d<rgba8unorm, write>;
// @group(1) @binding(0)
// var<storage, read> agents: array<Agent>;

@compute @workgroup_size(1)
fn draw_agents(@builtin(global_invocation_id) id: vec3<u32>) {
	let dims = textureDimensions(agent_texture);
	
	if agents[id.x].x < f32(dims.x) || agents[id.x].x >= 0.0 || agents[id.x].y < f32(dims.y) || agents[id.x].y >= 0.0 { 
		textureStore(agent_texture, vec2<i32>(i32(agents[id.x].x), i32(agents[id.x].y)), vec4<f32>(1.0, 1.0, 1.0, 1.0));
	}
}

@compute @workgroup_size(1, 1)
fn dim_texture(@builtin(global_invocation_id) id: vec3<u32>) {
	var color: vec4<f32> = textureLoad(agent_texture, vec2<i32>(id.xy));
	color.x -= min(color.x, 0.01);
	color.y -= min(color.y, 0.01);
	color.z -= min(color.z, 0.01);

	textureStore(agent_texture, vec2<i32>(id.xy), color);
}
