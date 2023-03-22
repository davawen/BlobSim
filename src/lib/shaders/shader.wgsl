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

fn in_bounds(dims: vec2<i32>, x: f32, y: f32) -> bool {
	return x < f32(dims.x) && x >= 0.0 && y < f32(dims.y) && y >= 0.0;
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
@group(0) @binding(1)
var agent_blur_texture: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(0)
var<storage, read_write> agents: array<Agent>;

@compute @workgroup_size(64)
fn update_agents(@builtin(global_invocation_id) id: vec3<u32>) {
	let PI: f32 = 3.14159265359;

	// Randomly steer
	agents[id.x].direction += f32(hash(u32(agents[id.x].direction * 500.0) + id.x)) / 4294967295.0 * (f32(hash(u32(agents[id.x].direction * 2000.0) + id.x)) / 4294967295.0 - 0.5)*0.1;

	let new_x = agents[id.x].x + cos(agents[id.x].direction);
	let new_y = agents[id.x].y + sin(agents[id.x].direction);

	let dims = textureDimensions(agent_texture);
	if !in_bounds(dims, new_x, new_y) { 
		agents[id.x].direction = f32(hash(u32(agents[id.x].direction * 1000.0) + id.x)) / 4294967295.0 * 2.0 * PI;
	}
	else {
		agents[id.x].x = new_x;
		agents[id.x].y = new_y;
	}
}

fn sense(agent_texture: texture_storage_2d<rgba8unorm, read_write>, pos: vec2<i32>) -> f32 {
	// 3 = total width
	// size = half width + center
	let size: i32 = 3 / 2;

	var amount = 0.0;
	for(var i: i32 = -size; i <= size; i++) {
		for(var j: i32 = -size; j <= size; j++) {
			amount += textureLoad(agent_texture, vec2<i32>(pos.x + i, pos.y + j)).r;
		}
	}
	return amount;
}

@compute @workgroup_size(64)
fn sense_agents(@builtin(global_invocation_id) id: vec3<u32>) {
	let PI: f32 = 3.14159265359;
	let angle = PI/3.0; // 60 degrees
	let strength = 0.2;
	let dist = 3.0;

	let agent = agents[id.x];

	let left = agent.direction - angle;
	let right = agent.direction + angle;

	let left   = vec2<f32>(agent.x + cos(left)*dist, agent.y + sin(left)*dist);
	let center = vec2<f32>(agent.x + cos(agent.direction)*dist, agent.y + sin(agent.direction)*dist);
	let right  = vec2<f32>(agent.x + cos(right)*dist, agent.y + sin(right)*dist);

	let dims = textureDimensions(agent_texture);

	var left_amount = sense(agent_texture, vec2<i32>(left));
	var center_amount = sense(agent_texture, vec2<i32>(center));
	var right_amount = sense(agent_texture, vec2<i32>(right));

	if left_amount > right_amount && left_amount > center_amount {
		agents[id.x].direction -= angle * strength;
	} else if right_amount > left_amount && right_amount > center_amount {
		agents[id.x].direction += angle * strength;
	}
}

// @group(0) @binding(0)
// var agent_texture: texture_storage_2d<rgba8unorm, write>;
// @group(1) @binding(0)
// var<storage, read> agents: array<Agent>;

@compute @workgroup_size(64)
fn draw_agents(@builtin(global_invocation_id) id: vec3<u32>) {
	let dims = textureDimensions(agent_texture);

	if in_bounds(dims, agents[id.x].x, agents[id.x].y) {
		textureStore(agent_texture, vec2<i32>(i32(agents[id.x].x), i32(agents[id.x].y)), vec4<f32>(1.0, 1.0, 1.0, 1.0));
	}
}

@compute @workgroup_size(10, 10)
fn dim_texture(@builtin(global_invocation_id) id: vec3<u32>) {
	var color: vec4<f32> = textureLoad(agent_texture, vec2<i32>(id.xy));
	color.x -= min(color.x, 0.02);
	color.y -= min(color.y, 0.02);
	color.z -= min(color.z, 0.02);

	textureStore(agent_texture, vec2<i32>(id.xy), color);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
	return a + (b-a)*t;
}

@compute @workgroup_size(10, 10)
fn blur_texture(@builtin(global_invocation_id) id: vec3<u32>) {
	let id = vec2<i32>(id.xy);

	let size = 1;
	var mean: f32 = 0.0;
	for(var i: i32 = -size; i <= size; i++) {
		for(var j: i32 = -size; j <= size; j++) {
			mean += textureLoad(agent_texture, vec2<i32>(max(id.x + i, 0), max(id.y + j, 0))).r;
		}
	}
	mean /= f32(size * 2 + 1)*f32(size * 2 + 1);

	let value = lerp(textureLoad(agent_texture, id).r, mean, 0.1);

	textureStore(agent_blur_texture, id, vec4<f32>(value, value, value, 1.0));
}
