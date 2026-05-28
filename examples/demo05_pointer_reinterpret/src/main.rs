use bytemuck::{Pod, Zeroable, cast_slice, cast_slice_mut, cast_ref};

/// Example 1: Vec3 with #[repr(C)] and bytemuck for safe reinterpretation
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    
    // Named accessors (no union needed - these are zero-cost)
    pub const fn a(&self) -> f64 { self.x }
    pub const fn b(&self) -> f64 { self.y }
    pub const fn c(&self) -> f64 { self.z }
    
    pub const fn i(&self) -> f64 { self.x }
    pub const fn j(&self) -> f64 { self.y }
    pub const fn k(&self) -> f64 { self.z }
    
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn norm(&self) -> f64 { self.dot(self).sqrt() }
}

/// Example 2: Mat3 with column/row views
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Mat3 {
    pub xx: f64, pub xy: f64, pub xz: f64,
    pub yx: f64, pub yy: f64, pub yz: f64,
    pub zx: f64, pub zy: f64, pub zz: f64,
}

impl Mat3 {
    pub fn a(&self) -> Vec3 { Vec3::new(self.xx, self.yx, self.zx) } // first column
    pub fn b(&self) -> Vec3 { Vec3::new(self.xy, self.yy, self.zy) } // second column
    pub fn c(&self) -> Vec3 { Vec3::new(self.xz, self.yz, self.zz) } // third column
}

/// Example 3: Quat4 with force/energy view
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Quat4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat4 {
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }
    pub fn f(&self) -> Vec3 { Vec3::new(self.x, self.y, self.z) }  // "force" or vector part
    pub fn e(&self) -> f64 { self.w }                               // "energy" or scalar part
}

/// Example 4: Trait for flat array view
pub trait AsFlatArray {
    const N: usize;
    fn as_flat(&self) -> &[f64];
    fn as_flat_mut(&mut self) -> &mut [f64];
}

impl AsFlatArray for Vec3 {
    const N: usize = 3;
    fn as_flat(&self) -> &[f64] { cast_slice(std::slice::from_ref(self)) }
    fn as_flat_mut(&mut self) -> &mut [f64] { cast_slice_mut(std::slice::from_mut(self)) }
}

impl AsFlatArray for Mat3 {
    const N: usize = 9;
    fn as_flat(&self) -> &[f64] { cast_slice(std::slice::from_ref(self)) }
    fn as_flat_mut(&mut self) -> &mut [f64] { cast_slice_mut(std::slice::from_mut(self)) }
}

impl AsFlatArray for Quat4 {
    const N: usize = 4;
    fn as_flat(&self) -> &[f64] { cast_slice(std::slice::from_ref(self)) }
    fn as_flat_mut(&mut self) -> &mut [f64] { cast_slice_mut(std::slice::from_mut(self)) }
}

fn main() {
    println!("=== Rust Pointer Type Reinterpretation Examples ===\n");

    // Example 1: Cast Vec3 slice to flat f64 slice
    println!("Example 1: Vec3 slice -> flat f64 slice");
    let positions = vec![
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    ];
    let flat: &[f64] = cast_slice(&positions);
    println!("  positions.len() = {}", positions.len());
    println!("  flat.len() = {}", flat.len());
    println!("  flat = {:?}", flat);
    println!("  positions[0].x = {}, flat[0] = {}", positions[0].x, flat[0]);
    println!("  positions[1].y = {}, flat[4] = {}", positions[1].y, flat[4]);
    println!();

    // Example 2: Mutable cast
    println!("Example 2: Mutable Vec3 slice -> mutable flat f64 slice");
    let mut positions_mut = vec![Vec3::new(0.0, 0.0, 0.0); 3];
    let flat_mut: &mut [f64] = cast_slice_mut(&mut positions_mut);
    flat_mut[0] = 1.0;
    flat_mut[1] = 2.0;
    flat_mut[2] = 3.0;
    println!("  After writing to flat_mut[0..2]: {:?}", positions_mut[0]);
    println!();

    // Example 3: Named accessors (C++ union replacement)
    println!("Example 3: Named accessors (C++ union replacement)");
    let v = Vec3::new(1.0, 2.0, 3.0);
    println!("  v.x = {}, v.a() = {}, v.i() = {}", v.x, v.a(), v.i());
    println!("  v.y = {}, v.b() = {}, v.j() = {}", v.y, v.b(), v.j());
    println!("  v.z = {}, v.c() = {}, v.k() = {}", v.z, v.c(), v.k());
    println!();

    // Example 4: Matrix column views
    println!("Example 4: Matrix column views");
    let m = Mat3 {
        xx: 1.0, xy: 2.0, xz: 3.0,
        yx: 4.0, yy: 5.0, yz: 6.0,
        zx: 7.0, zy: 8.0, zz: 9.0,
    };
    println!("  m.a() = {:?}", m.a());
    println!("  m.b() = {:?}", m.b());
    println!("  m.c() = {:?}", m.c());
    println!();

    // Example 5: Quaternion force/energy view
    println!("Example 5: Quaternion force/energy view");
    let q = Quat4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
    println!("  q.f() = {:?}", q.f());
    println!("  q.e() = {}", q.e());
    println!();

    // Example 6: Generic function using trait
    println!("Example 6: Generic function using AsFlatArray trait");
    fn print_flat<T: AsFlatArray>(item: &T, name: &str) {
        let flat = item.as_flat();
        println!("  {} as flat: {:?}", name, flat);
    }
    print_flat(&Vec3::new(1.0, 2.0, 3.0), "Vec3");
    print_flat(&Mat3::default(), "Mat3");
    print_flat(&Quat4::new(1.0, 2.0, 3.0, 4.0), "Quat4");
    println!();

    // Example 7: Forcefield-style kernel with flat arrays
    println!("Example 7: Forcefield-style kernel with flat arrays");
    let positions = vec![Vec3::new(0.0, 0.0, 0.0); 3];
    let mut forces = vec![Vec3::new(0.0, 0.0, 0.0); 3];
    let _masses = vec![1.0, 2.0, 3.0];
    
    // Cast to flat arrays for the kernel
    let _pos_flat: &[f64] = cast_slice(&positions);
    let force_flat: &mut [f64] = cast_slice_mut(&mut forces);
    
    // Simple kernel: apply constant force
    for i in 0..force_flat.len() {
        force_flat[i] = 0.1;
    }
    
    println!("  After applying forces: {:?}", forces);
    println!();

    // Example 8: Cast single struct to array
    println!("Example 8: Cast single struct to array");
    let v = Vec3::new(1.0, 2.0, 3.0);
    let arr: &[f64; 3] = cast_ref(&v);
    println!("  Vec3 as array: {:?}", arr);
    println!();

    println!("=== All examples completed successfully! ===");
}
