use num_complex::Complex64;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

use rustacuda::context::{Context, ContextFlags, CurrentContext};
use rustacuda::device::Device;
use rustacuda::memory::{DeviceBuffer, DevicePointer};
use rustacuda::module::Module;
use rustacuda::prelude::*;
use rustacuda::stream::{Stream, StreamFlags};

struct Circuit {
    num_qubits: usize,
    gates: Vec<(Gate, Vec<usize>)>,
}

impl Circuit {
    fn new(num_qubits: usize) -> Self {
        Circuit {
            num_qubits,
            gates: Vec::new(),
        }
    }

    fn add_gate(&mut self, gate: Gate, targets: Vec<usize>) {
        self.gates.push((gate, targets));
    }

    fn run(&self, use_gpu: bool) -> Vec<Complex64> {
        let mut state = vec![Complex64::new(0.0, 0.0); 1 << self.num_qubits];
        state[0] = Complex64::new(1.0, 0.0);
        for (gate, targets) in &self.gates {
            state = gate.apply(&state, targets.clone(), self.num_qubits, use_gpu);
        }
        state
    }

    fn compute_probabilities(&self, state: &[Complex64]) -> Vec<f64> {
        let norm: f64 = state.iter().map(|amp| amp.norm_sqr()).sum();

        if norm.abs() < 1e-12 {
            eprintln!("error: state norm is too small probabilities cannot be computed");
            return vec![0.0; state.len()];
        }

        let normalized_state: Vec<Complex64> = state.iter().map(|amp| *amp / norm.sqrt()).collect();
        normalized_state.iter().map(|amp| amp.norm_sqr()).collect()
    }
}

#[derive(Clone, Debug)]
enum Gate {
    H,
    T,
    X,
    Y,
    Z,
    ID,
    CNOT,
    SWAP,
}

impl Gate {
    fn apply(
        &self,
        state: &[Complex64],
        targets: Vec<usize>,
        num_qubits: usize,
        use_gpu: bool,
    ) -> Vec<Complex64> {
        match self {
            Gate::H => apply_single_qubit_gate(state, hadamard(), targets[0], num_qubits, use_gpu),
            Gate::T => apply_single_qubit_gate(state, t_gate(), targets[0], num_qubits, use_gpu),
            Gate::X => apply_single_qubit_gate(state, pauli_x(), targets[0], num_qubits, use_gpu),
            Gate::Y => apply_single_qubit_gate(state, pauli_y(), targets[0], num_qubits, use_gpu),
            Gate::Z => apply_single_qubit_gate(state, pauli_z(), targets[0], num_qubits, use_gpu),
            Gate::ID => apply_single_qubit_gate(state, identity(), targets[0], num_qubits, use_gpu),
            Gate::CNOT => apply_two_qubit_gate(state, cnot(), targets, num_qubits),
            Gate::SWAP => apply_two_qubit_gate(state, swap(), targets, num_qubits),
        }
    }
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let scale = 1.0 / (2.0_f64).sqrt();
    [
        [Complex64::new(scale, 0.0), Complex64::new(scale, 0.0)],
        [Complex64::new(scale, 0.0), Complex64::new(-scale, 0.0)],
    ]
}

fn t_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.7071, 0.7071)],
    ]
}

fn pauli_x() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_y() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_z() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]
}

fn identity() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ]
}

fn cnot() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ]
}

fn swap() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ]
}

//gpu
fn apply_single_qubit_gate(
    state: &[Complex64],
    gate: [[Complex64; 2]; 2],
    target: usize,
    num_qubits: usize,
    use_gpu: bool,
) -> Vec<Complex64> {
    if !use_gpu {
        //cpu
        let start = Instant::now();
        let dim = 1 << num_qubits;
        let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

        for i in 0..dim {
            let target_bit = (i >> target) & 1;
            for j in 0..2 {
                let source = (i & !(1 << target)) | (j << target);
                new_state[i] += gate[target_bit][j] * state[source];
            }
        }
        let duration = start.elapsed();
        println!("CPU gate application time: {:?}", duration);
        return new_state;
    } else {
        //gpu
        let start = Instant::now();
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _context = Context::create_and_push(ContextFlags::MAP_HOST, device).unwrap();

        //load fat-bin
        let module_data = include_bytes!("kernels.fatbin");
        let module = Module::load_from_bytes(module_data).unwrap();
        let function = module
            .get_function("apply_single_qubit_gate_kernel")
            .unwrap();

        let dim = 1 << num_qubits;
        let mut re_in = Vec::with_capacity(dim);
        let mut im_in = Vec::with_capacity(dim);
        let mut re_gate = Vec::new();
        let mut im_gate = Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                re_gate.push(gate[row][col].re);
                im_gate.push(gate[row][col].im);
            }
        }

        for amp in state {
            re_in.push(amp.re);
            im_in.push(amp.im);
        }

        let mut re_out = vec![0.0; dim];
        let mut im_out = vec![0.0; dim];

        let mut d_re_in = DeviceBuffer::from_slice(&re_in).unwrap();
        let mut d_im_in = DeviceBuffer::from_slice(&im_in).unwrap();
        let mut d_re_out = DeviceBuffer::from_slice(&re_out).unwrap();
        let mut d_im_out = DeviceBuffer::from_slice(&im_out).unwrap();
        let d_re_gate = DeviceBuffer::from_slice(&re_gate).unwrap();
        let d_im_gate = DeviceBuffer::from_slice(&im_gate).unwrap();

        let block_size = 128;
        let grid_size = (dim as u32 + block_size - 1) / block_size;

        let mut stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

        unsafe {
            launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    d_re_in.as_device_ptr(),
                    d_im_in.as_device_ptr(),
                    d_re_out.as_device_ptr(),
                    d_im_out.as_device_ptr(),
                    d_re_gate.as_device_ptr(),
                    d_im_gate.as_device_ptr(),
                    target as i32,
                    num_qubits as i32
                )
            )
            .unwrap();
        }

        stream.synchronize().unwrap();

        d_re_out.copy_to(&mut re_out).unwrap();
        d_im_out.copy_to(&mut im_out).unwrap();

        let mut new_state = vec![Complex64::new(0.0, 0.0); dim];
        for i in 0..dim {
            new_state[i] = Complex64::new(re_out[i], im_out[i]);
        }
        let duration = start.elapsed();
        println!("GPU gate application time: {:?}", duration);
        new_state
    }
}

fn apply_two_qubit_gate(
    state: &[Complex64],
    gate: [[Complex64; 4]; 4],
    targets: Vec<usize>,
    num_qubits: usize,
) -> Vec<Complex64> {
    // leave two-qubit gate application on CPU
    let dim = 1 << num_qubits;
    let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

    let control = targets[0];
    let target = targets[1];
    for i in 0..dim {
        let control_bit = (i >> control) & 1;
        let target_bit = (i >> target) & 1;
        let index = (control_bit << 1) | target_bit;
        for j in 0..4 {
            let source = (i & !(1 << control) & !(1 << target))
                | ((j >> 1) << control)
                | ((j & 1) << target);
            new_state[i] += gate[index][j] * state[source];
        }
    }
    new_state
}

fn main() {
    println!("Enter the number of qubits:");
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");
    let num_qubits: usize = input.trim().parse().expect("Invalid number of qubits");

    if num_qubits == 0 {
        eprintln!("Error: Number of qubits must be greater than 0.");
        return;
    }

    let mut circuit = Circuit::new(num_qubits);

    let gate_map: HashMap<&str, Gate> = [
        ("h", Gate::H),
        ("t", Gate::T),
        ("x", Gate::X),
        ("y", Gate::Y),
        ("z", Gate::Z),
        ("id", Gate::ID),
        ("cnot", Gate::CNOT),
        ("swap", Gate::SWAP),
    ]
    .iter()
    .map(|&(k, ref v)| (k, v.clone()))
    .collect();

    println!("Enter your circuit instructions:");
    println!("Type 'run' when finished:");

    loop {
        print!("-> ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let instruction = input.trim();

        if instruction.eq_ignore_ascii_case("run") {
            break;
        }

        let parts: Vec<&str> = instruction.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Invalid instruction. Example: 'h q[0]' or 'cnot q[0],q[1]'");
            continue;
        }

        let gate_name = parts[0];
        let target_str = parts[1];

        let gate = match gate_map.get(gate_name) {
            Some(g) => g.clone(),
            None => {
                println!(
                    "Unknown gate: '{}'. Available gates: {:?}",
                    gate_name,
                    gate_map.keys()
                );
                continue;
            }
        };

        let targets: Vec<usize> = target_str
            .split(&['[', ']', ','])
            .filter_map(|s| s.parse::<usize>().ok())
            .collect();

        if targets.is_empty() || targets.iter().any(|&q| q >= num_qubits) {
            println!(
                "Invalid target qubits for instruction: '{}'. Max qubit index is {}.",
                instruction,
                num_qubits - 1
            );
            continue;
        }

        println!("Adding gate: {:?}, targets: {:?}", gate, targets);
        circuit.add_gate(gate, targets);
    }

    //cpu
    println!("Running the circuit on CPU...");
    let final_state_cpu = circuit.run(false);

    //gpu
    println!("Running the circuit on GPU...");
    let final_state_gpu = circuit.run(true);

    println!("Final state vector (CPU):");
    for (index, amplitude) in final_state_cpu.iter().enumerate() {
        println!("|{:0width$b}>: {}", index, amplitude, width = num_qubits);
    }

    println!("Final state vector (GPU):");
    for (index, amplitude) in final_state_gpu.iter().enumerate() {
        println!("|{:0width$b}>: {}", index, amplitude, width = num_qubits);
    }

    let probabilities_cpu = circuit.compute_probabilities(&final_state_cpu);
    let probabilities_gpu = circuit.compute_probabilities(&final_state_gpu);

    println!("Probabilities (CPU):");
    for (state, prob) in probabilities_cpu.iter().enumerate() {
        if *prob > 0.0 {
            println!(
                "State |{:0width$b}>: {:.5}",
                state,
                prob,
                width = num_qubits
            );
        }
    }

    println!("Probabilities (GPU):");
    for (state, prob) in probabilities_gpu.iter().enumerate() {
        if *prob > 0.0 {
            println!(
                "State |{:0width$b}>: {:.5}",
                state,
                prob,
                width = num_qubits
            );
        }
    }
}

/*use num_complex::Complex64;
use std::collections::HashMap;
use std::io::{self, Write};

struct Circuit {
    num_qubits: usize,
    gates: Vec<(Gate, Vec<usize>)>,
}

impl Circuit {
    fn new(num_qubits: usize) -> Self {
        Circuit {
            num_qubits,
            gates: Vec::new(),
        }
    }

    fn add_gate(&mut self, gate: Gate, targets: Vec<usize>) {
        self.gates.push((gate, targets));
    }

    fn run(&self) -> Vec<Complex64> {
        let mut state = vec![Complex64::new(0.0, 0.0); 1 << self.num_qubits];
        state[0] = Complex64::new(1.0, 0.0); // Initialize state |0...0>
        for (gate, targets) in &self.gates {
            state = gate.apply(&state, targets.clone(), self.num_qubits);
        }
        state
    }

    fn compute_probabilities(&self, state: &[Complex64]) -> Vec<f64> {
        // Compute the total norm of the state
        let norm: f64 = state.iter().map(|amp| amp.norm_sqr()).sum();

        if norm.abs() < 1e-12 {
            eprintln!("Error: State norm is too small! Probabilities cannot be computed.");
            return vec![0.0; state.len()];
        }

        // Normalize the state
        let normalized_state: Vec<Complex64> = state.iter().map(|amp| *amp / norm.sqrt()).collect();

        // Compute probabilities from the normalized state
        normalized_state.iter().map(|amp| amp.norm_sqr()).collect()
    }
}

#[derive(Clone, Debug)]
enum Gate {
    H,
    T,
    X,
    Y,
    Z,
    ID,
    CNOT,
    SWAP,
}

impl Gate {
    fn apply(&self, state: &[Complex64], targets: Vec<usize>, num_qubits: usize) -> Vec<Complex64> {
        match self {
            Gate::H => apply_single_qubit_gate(state, hadamard(), targets[0], num_qubits),
            Gate::T => apply_single_qubit_gate(state, t_gate(), targets[0], num_qubits),
            Gate::X => apply_single_qubit_gate(state, pauli_x(), targets[0], num_qubits),
            Gate::Y => apply_single_qubit_gate(state, pauli_y(), targets[0], num_qubits),
            Gate::Z => apply_single_qubit_gate(state, pauli_z(), targets[0], num_qubits),
            Gate::ID => apply_single_qubit_gate(state, identity(), targets[0], num_qubits),
            Gate::CNOT => apply_two_qubit_gate(state, cnot(), targets, num_qubits),
            Gate::SWAP => apply_two_qubit_gate(state, swap(), targets, num_qubits),
        }
    }
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let scale = 1.0 / (2.0_f64).sqrt();
    [
        [Complex64::new(scale, 0.0), Complex64::new(scale, 0.0)],
        [Complex64::new(scale, 0.0), Complex64::new(-scale, 0.0)],
    ]
}

fn t_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.7071, 0.7071)],
    ]
}

fn pauli_x() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_y() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_z() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]
}

fn identity() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ]
}

fn cnot() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ]
}

fn swap() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ]
}

fn apply_single_qubit_gate(
    state: &[Complex64],
    gate: [[Complex64; 2]; 2],
    target: usize,
    num_qubits: usize,
) -> Vec<Complex64> {
    let dim = 1 << num_qubits;
    let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

    for i in 0..dim {
        let target_bit = (i >> target) & 1; // Extract the target bit
        for j in 0..2 {
            let source = (i & !(1 << target)) | (j << target); // Flip the target bit
            new_state[i] += gate[target_bit][j] * state[source];
        }
    }

    // Debugging: print intermediate states for verification
    //println!("State after single-qubit gate: {:?}", new_state);

    new_state
}

fn apply_two_qubit_gate(
    state: &[Complex64],
    gate: [[Complex64; 4]; 4],
    targets: Vec<usize>,
    num_qubits: usize,
) -> Vec<Complex64> {
    let dim = 1 << num_qubits;
    let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

    let control = targets[0];
    let target = targets[1];

    for i in 0..dim {
        let control_bit = (i >> control) & 1; // Extract control bit
        let target_bit = (i >> target) & 1; // Extract target bit

        let index = (control_bit << 1) | target_bit; // Combine control and target bits
        for j in 0..4 {
            let source = (i & !(1 << control) & !(1 << target))
                | ((j >> 1) << control)  // Set the new control bit
                | ((j & 1) << target); // Set the new target bit
            new_state[i] += gate[index][j] * state[source];
        }
    }

    // Debugging: print intermediate states for verification
    //println!("State after two-qubit gate: {:?}", new_state);

    new_state
}

fn main() {
    println!("Enter the number of qubits:");
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");
    let num_qubits: usize = input.trim().parse().expect("Invalid number of qubits");

    if num_qubits == 0 {
        eprintln!("Error: Number of qubits must be greater than 0.");
        return;
    }

    let mut circuit = Circuit::new(num_qubits);

    let gate_map: HashMap<&str, Gate> = [
        ("h", Gate::H),
        ("t", Gate::T),
        ("x", Gate::X),
        ("y", Gate::Y),
        ("z", Gate::Z),
        ("id", Gate::ID),
        ("cnot", Gate::CNOT),
        ("swap", Gate::SWAP),
    ]
    .iter()
    .map(|&(k, ref v)| (k, v.clone()))
    .collect();

    println!("Enter your circuit instructions:");
    println!("Type 'run' when finished:");

    loop {
        print!("-> ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let instruction = input.trim();

        if instruction.eq_ignore_ascii_case("run") {
            break;
        }

        let parts: Vec<&str> = instruction.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Invalid instruction. Example: 'h q[0]' or 'cnot q[0],q[1]'");
            continue;
        }

        let gate_name = parts[0];
        let target_str = parts[1];

        let gate = match gate_map.get(gate_name) {
            Some(g) => g.clone(),
            None => {
                println!(
                    "Unknown gate: '{}'. Available gates: {:?}",
                    gate_name,
                    gate_map.keys()
                );
                continue;
            }
        };

        let targets: Vec<usize> = target_str
            .split(&['[', ']', ','])
            .filter_map(|s| s.parse::<usize>().ok())
            .collect();

        if targets.is_empty() || targets.iter().any(|&q| q >= num_qubits) {
            println!(
                "Invalid target qubits for instruction: '{}'. Max qubit index is {}.",
                instruction,
                num_qubits - 1
            );
            continue;
        }

        println!("Adding gate: {:?}, targets: {:?}", gate, targets);
        circuit.add_gate(gate, targets);
    }

    // Run the circuit
    println!("Running the circuit...");
    let final_state = circuit.run();

    // Debug: Print the final state vector
    println!("Final state vector:");
    for (index, amplitude) in final_state.iter().enumerate() {
        println!("|{:0width$b}>: {}", index, amplitude, width = num_qubits);
    }

    // Compute and validate probabilities
    let probabilities = circuit.compute_probabilities(&final_state);
    let total_prob: f64 = probabilities.iter().sum();

    if (total_prob - 1.0).abs() > 1e-6 {
        eprintln!(
            "Warning: Probabilities do not sum to 1! Total = {:.6}",
            total_prob
        );
    } else {
        println!("Probabilities are normalized. Total = {:.6}", total_prob);
    }

    // Output probabilities
    println!("Probabilities of each state:");
    for (state, prob) in probabilities.iter().enumerate() {
        if *prob > 0.0 {
            println!(
                "State |{:0width$b}>: {:.5}",
                state,
                prob,
                width = num_qubits
            );
        }
    }
}
*/
