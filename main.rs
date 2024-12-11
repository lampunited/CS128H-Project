use num_complex::Complex64;
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
        println!("Adding gate {:?} on qubits {:?}", gate, targets);
        self.gates.push((gate, targets));
    }

    fn run(&self) -> Vec<Complex64> {
        let mut state = vec![Complex64::new(0.0, 0.0); 1 << self.num_qubits];
        state[0] = Complex64::new(1.0, 0.0);
        for (gate, targets) in &self.gates {
            println!("Applying {:?} on {:?}", gate, targets);
            state = gate.apply(&state, targets.clone(), self.num_qubits);
        }
        state
    }

    fn compute_probabilities(&self, state: &[Complex64]) -> Vec<f64> {
        state.iter().map(|amp| amp.norm_sqr()).collect()
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
        let target_bit = (i >> target) & 1;
        for j in 0..2 {
            let source = (i & !(1 << target)) | (j << target);
            new_state[i] += gate[target_bit][j] * state[source];
        }
    }

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
    let mut input = String::new();
    print!("Enter number of qubits: ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    let num_qubits: usize = input.trim().parse().expect("Invalid number");

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

    print!("Enter number of instructions: ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    let num_instructions: usize = input.trim().parse().expect("Invalid number");

    for _ in 0..num_instructions {
        input.clear();
        print!("Enter instruction (e.g. 'h q[0]' or 'x q[1]'): ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        let instruction = input.trim();

        let parts: Vec<&str> = instruction.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Invalid instruction: {}", instruction);
            continue;
        }

        let gate_name = parts[0];
        let target_str = parts[1];

        let gate = match gate_map.get(gate_name) {
            Some(g) => g.clone(),
            None => {
                println!("Unknown gate: {}", gate_name);
                continue;
            }
        };

        let targets: Vec<usize> = target_str
            .split(&['[', ']', ','])
            .filter_map(|s| s.parse::<usize>().ok())
            .collect();

        if targets.is_empty() || targets.iter().any(|&q| q >= num_qubits) {
            println!("Invalid target qubits for instruction: {}", instruction);
            continue;
        }

        circuit.add_gate(gate, targets);
    }

    println!("Starting circuit execution...");
    let final_state = circuit.run();

    let probabilities = circuit.compute_probabilities(&final_state);
    println!("Final probabilities:");
    for (state, prob) in probabilities.iter().enumerate() {
        println!(
            "State |{:0width$b}>: {:.5}",
            state,
            prob,
            width = num_qubits
        );
    }
}
