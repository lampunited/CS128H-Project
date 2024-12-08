# CS128H-Project

---Group Name: Quantum Circuits

Sushanth - se13

Steven - stevenp8

Yotam - dubiner2

Edward - edwardh6

---Project Introduction

The project’s goal is to build a basic quantum circuit simulator in Rust, allowing for simulations of essential quantum gates and simple algorithms like Grover’s search. We’ll keep it light on advanced features, focusing on developing an understanding of quantum computing concepts and getting comfortable with Rust and CUDA basics for potential speed improvements.

---Technical Overview

We’ll tackle the project in four steps:

-Basic Quantum Gates

Build simple gates (Hadamard, Pauli-X, CNOT) that can operate on single qubits and simple quantum states.

Tasks: Learn how to represent a qubit, apply basic gate functions, and see results on single qubits.

-Grover’s Algorithm

Implement Grover’s algorithm to gain experience with multi-step quantum operations.

Tasks: Set up a basic sequence of gates in Grover’s algorithm and test it with simple, predefined states.

-CUDA Exploration for Speeding Up Calculations

Explore using CUDA to handle larger matrix calculations for simulating quantum states. Focus only on one or two functions, like matrix multiplication, to keep it manageable.

Tasks: Set up CUDA for Rust, create a simple CUDA function to accelerate a matrix operation, and test it out.

-Simple Input and Output

Allow basic user inputs for defining a circuit and display outputs.

Tasks: Make a straightforward input/output for adding gates and seeing the state transformations.

---Project Schedule and Goals

-Checkpoint 1 (Weeks 1-2):

Understand Rust basics, set up qubits, and build initial quantum gates.

Test each gate function to see it work on single qubits.

-Checkpoint 2 (Weeks 3-4):

Implement Grover’s algorithm, testing each step separately for simplicity.

Keep Grover’s limited to a few qubits to make testing easier.

-Checkpoint 3 (Weeks 5-6):

Experiment with CUDA: Set up a simple matrix multiplication function with CUDA, if time allows.

Run basic tests with CUDA to see if calculations run faster.

-Final Checkpoint (Weeks 7-8):

Complete input/output setup, focusing on keeping it clear and simple.

Finalize and test each feature, making sure all major components work together.

---Possible Challenges

Learning Rust: Some Rust syntax, especially memory management, can be tricky at first.

CUDA Basics: Setting up Rust to work with CUDA may take time, and error handling could be challenging.

Quantum Concepts: Getting used to quantum operations like superposition and entanglement might need extra study.

---References

Resources like Qiskit’s online documentation for basic quantum concepts.

NVIDIA’s beginner CUDA guides.

Simple Rust quantum projects on GitHub to see how others approached similar challenges.

Test
Hello
Hello
Hello
