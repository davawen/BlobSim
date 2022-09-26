mod lib;
mod texture;

use crate::lib::run;

fn main() {
    pollster::block_on(run());
}
