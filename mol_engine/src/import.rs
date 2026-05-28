use std::path::Path;

use mol_topology::export::import_json;
use mol_topology::topology::Topology;

use crate::uff::Uff;

/// Load topology from JSON file and create UFF engine
pub fn load_topology_from_json<P: AsRef<Path>>(path: P) -> Result<(Uff, Vec<String>), Box<dyn std::error::Error>> {
    let (topology, elements) = import_json(path)?;
    let mut ff = Uff::from_topology(&topology);
    
    // Initialize neighbor structures
    ff.make_neigh_bs();
    ff.bake_angle_neighs();
    ff.bake_dihedral_neighs();
    ff.bake_inversion_neighs();
    ff.map_atom_interactions();
    
    Ok((ff, elements))
}
