# MPI and RDMA Programming

*Last Updated: May 25, 2025*

## Overview

Message Passing Interface (MPI) and Remote Direct Memory Access (RDMA) are critical technologies for high-performance computing and distributed systems. This learning track covers the principles, programming models, and optimization techniques for MPI and RDMA, focusing on implementing efficient parallel and distributed applications.

## Learning Path

### 1. MPI Fundamentals (2 weeks)
[See details in 01_MPI_Fundamentals.md](02_MPI_RDMA/01_MPI_Fundamentals.md)
- **MPI Overview**
  - History and evolution
  - Implementation standards (MPI-1, MPI-2, MPI-3, MPI-4)
  - Open source implementations (OpenMPI, MPICH)
  - Vendor implementations (Intel MPI, IBM Spectrum MPI)
- **MPI Programming Model**
  - SPMD (Single Program Multiple Data) concept
  - Process ranks and communicators
  - Basic execution model
  - MPI runtime environments
- **Basic Point-to-Point Communication**
  - Blocking sends and receives
  - Send modes (standard, synchronous, buffered, ready)
  - Message matching
  - Deadlock avoidance
- **Environment Management**
  - Initialization and finalization
  - Query functions
  - Error handling
  - Process naming

### 2. Advanced MPI Point-to-Point Communication (2 weeks)
[See details in 02_Advanced_MPI_Point-to-Point_Communication.md](02_MPI_RDMA/02_Advanced_MPI_Point-to-Point_Communication.md)
- **Non-blocking Communication**
  - MPI_Isend and MPI_Irecv
  - Request objects
  - Completion functions (wait, test)
  - Wait/test variants (any, some, all)
- **Persistent Communication**
  - Persistent request creation
  - Start and complete operations
  - Use cases and benefits
- **Probe-based Messaging**
  - MPI_Probe and MPI_Iprobe
  - Message extraction
  - Dynamic message sizes
- **Communication Modes**
  - Performance implications
  - Protocol selection
  - Eager vs. rendezvous protocols
  - Memory requirements

### 3. MPI Collective Communication (2 weeks)
[See details in 03_MPI_Collective_Communication.md](02_MPI_RDMA/03_MPI_Collective_Communication.md)
- **Basic Collective Operations**
  - Barriers (MPI_Barrier)
  - Broadcast (MPI_Bcast)
  - Scatter and gather (MPI_Scatter, MPI_Gather)
  - All-gather and all-to-all (MPI_Allgather, MPI_Alltoall)
  - Reductions (MPI_Reduce, MPI_Allreduce)
- **Advanced Collective Operations**
  - Vector variants (MPI_Scatterv, MPI_Gatherv)
  - Reduction operations
  - Scan and exclusive scan (MPI_Scan, MPI_Exscan)
  - User-defined operations
- **Non-blocking Collectives**
  - Non-blocking collective interfaces
  - Overlapping computation and communication
  - Progress considerations
- **Optimization Techniques**
  - Algorithm selection
  - Tuning parameters
  - Topology awareness
  - Hardware acceleration

### 4. MPI Derived Datatypes (1 week)
[See details in 04_MPI_Derived_Datatypes.md](02_MPI_RDMA/04_MPI_Derived_Datatypes.md)
- **Basic Type Construction**
  - Contiguous types
  - Vector types
  - Indexed types
  - Struct types
- **Type Manipulation**
  - Type commits and frees
  - Type duplication
  - Type resizing
  - Type extents and bounds
- **Advanced Type Features**
  - Subarray creation
  - Distributed array support
  - Type packing/unpacking
  - Type debugging
- **Performance Considerations**
  - Implementation-specific optimizations
  - Cache and memory alignment
  - Communication protocol interaction
  - Hardware acceleration

### 5. MPI Process Topology and Groups (1 week)
[See details in 05_MPI_Process_Topology_and_Groups.md](02_MPI_RDMA/05_MPI_Process_Topology_and_Groups.md)
- **Communicator Management**
  - Communicator creation
  - Communicator splitting
  - Intercommunicators
  - Communicator attributes
- **Process Groups**
  - Group creation and manipulation
  - Group operations (union, intersection, difference)
  - Group comparison
  - Rank translation
- **Cartesian Topologies**
  - Cartesian communicator creation
  - Coordinate-based ranking
  - Shift operations
  - Periodic boundaries
- **Graph Topologies**
  - Graph communicator creation
  - Nearest neighbors
  - Distributed graph topologies
  - Custom topology implementations

### 6. Advanced MPI Features (2 weeks)
[See details in 06_Advanced_MPI_Features.md](02_MPI_RDMA/06_Advanced_MPI_Features.md)
- **One-sided Communication (RMA)**
  - Memory windows
  - Put, get, and accumulate operations
  - Synchronization models (fence, PSCW, locks)
  - Atomic operations
- **MPI I/O**
  - File handling
  - Collective and individual I/O
  - File views
  - Hints and optimizations
- **Dynamic Process Management**
  - Spawning processes
  - Connecting/accepting connections
  - Name publishing
  - Universe size
- **Thread Support**
  - Threading levels
  - Thread safety considerations
  - Hybrid MPI+OpenMP programming
  - Progress threads

### 7. RDMA Fundamentals (2 weeks)
[See details in 07_RDMA_Fundamentals.md](02_MPI_RDMA/07_RDMA_Fundamentals.md)
- **RDMA Concept and Architecture**
  - Zero-copy data transfer
  - Kernel bypass
  - Transport protocols (IB, RoCE, iWARP)
  - Comparison with traditional networking
- **Hardware Components**
  - Host Channel Adapters (HCAs)
  - Switches and routers
  - Network Interface Cards (NICs)
  - RDMA-capable devices
- **RDMA Operations**
  - RDMA Read
  - RDMA Write
  - Atomic operations
  - Send/Receive semantics
- **Queue Pair Model**
  - Work Queues
  - Completion Queues
  - Event Channels
  - Queue Pair states

### 8. RDMA Programming Interfaces (2 weeks)
[See details in 08_RDMA_Programming_Interfaces.md](02_MPI_RDMA/08_RDMA_Programming_Interfaces.md)
- **Verbs API**
  - Connection management
  - Memory registration
  - Queue Pair operations
  - Completion handling
- **libfabric/OFI**
  - Endpoint types
  - Domain management
  - Memory regions
  - Asynchronous operations
- **UCX Framework**
  - Transport selection
  - Connection establishment
  - Memory handling
  - Tag matching
- **RDMA CM (Connection Manager)**
  - Address resolution
  - Connection establishment
  - Event handling
  - Connection teardown

### 9. RDMA Communication Patterns (2 weeks)
[See details in 09_RDMA_Communication_Patterns.md](02_MPI_RDMA/09_RDMA_Communication_Patterns.md)
- **Two-sided Communication**
  - Send/Receive semantics
  - Scatter/Gather operations
  - Immediate data
  - Signaled completions
- **One-sided Communication**
  - RDMA Read operations
  - RDMA Write operations
  - RDMA Write with immediate
  - Fence and ordering
- **Atomic Operations**
  - Compare-and-swap
  - Fetch-and-add
  - Ordering guarantees
  - Synchronization patterns
- **Shared Receive Queues**
  - SRQ creation and management
  - Work request distribution
  - Completion handling
  - XRC (eXtended Reliable Connection)

### 10. GPU-Accelerated MPI and RDMA (2 weeks)
[See details in 10_GPU-Accelerated_MPI_and_RDMA.md](02_MPI_RDMA/10_GPU-Accelerated_MPI_and_RDMA.md)
- **CUDA-Aware MPI**
  - Direct GPU buffer access
  - Implementation specifics
  - Performance considerations
  - Hardware requirements
- **GPU Direct RDMA**
  - Architecture overview
  - Direct GPU-to-GPU transfers
  - GPU memory registration
  - CUDA integration
- **ROCm-Aware MPI**
  - AMD GPU integration
  - HIP programming model
  - Memory handling
  - Performance optimization
- **Performance Optimization**
  - Data locality
  - Computation-communication overlap
  - GPU kernel scheduling
  - Multi-GPU systems

### 11. Performance Analysis and Optimization (2 weeks)
[See details in 11_Performance_Analysis_and_Optimization.md](02_MPI_RDMA/11_Performance_Analysis_and_Optimization.md)
- **MPI Performance Analysis**
  - MPI profiling interface (PMPI)
  - Profiling tools (mpiP, TAU, Scalasca)
  - Trace collection and analysis
  - Performance metrics
- **RDMA Performance Analysis**
  - Performance counters
  - Latency and bandwidth measurement
  - Congestion analysis
  - QoS considerations
- **Optimization Techniques**
  - Message size optimization
  - Protocol selection
  - Memory registration strategies
  - Computation-communication overlap
- **Scalability Analysis**
  - Strong and weak scaling
  - Communication patterns
  - Collective algorithm selection
  - Network topology considerations

### 12. Advanced Applications and Case Studies (1 week)
[See details in 12_Advanced_Applications_and_Case_Studies.md](02_MPI_RDMA/12_Advanced_Applications_and_Case_Studies.md)
- **HPC Applications**
  - Scientific computing
  - Computational fluid dynamics
  - Molecular dynamics
  - Weather modeling
- **Big Data Processing**
  - Distributed analytics
  - In-memory databases
  - Stream processing
  - MapReduce implementations
- **Machine Learning**
  - Distributed training
  - Parameter servers
  - AllReduce algorithms
  - Model parallelism
- **Storage Systems**
  - Parallel file systems
  - Distributed object stores
  - RDMA-accelerated storage
  - Key-value stores

## Projects

1. **Distributed Matrix Multiplication**
   [See project details in project_01_Distributed_Matrix_Multiplication.md](02_MPI_RDMA/project_01_Distributed_Matrix_Multiplication.md)
   - Implement parallel matrix multiplication with MPI
   - Optimize communication patterns
   - Benchmark against sequential implementation

2. **RDMA-based Key-Value Store**
   [See project details in project_02_RDMA-based_Key-Value_Store.md](02_MPI_RDMA/project_02_RDMA-based_Key-Value_Store.md)
   - Create a simple key-value store using RDMA
   - Implement both two-sided and one-sided operations
   - Benchmark performance characteristics

3. **GPU-Accelerated N-Body Simulation**
   [See project details in project_03_GPU-Accelerated_N-Body_Simulation.md](02_MPI_RDMA/project_03_GPU-Accelerated_N-Body_Simulation.md)
   - Develop an N-body simulation using CUDA and MPI
   - Use GPU Direct RDMA for inter-node communication
   - Analyze scaling behavior

4. **Custom Collective Operation**
   [See project details in project_04_Custom_Collective_Operation.md](02_MPI_RDMA/project_04_Custom_Collective_Operation.md)
   - Implement a specialized collective algorithm
   - Optimize for specific network topologies
   - Compare with standard MPI collectives

5. **High-Performance Data Pipeline**
   [See project details in project_05_High-Performance_Data_Pipeline.md](02_MPI_RDMA/project_05_High-Performance_Data_Pipeline.md)
   - Build a data processing pipeline with RDMA
   - Implement zero-copy data flow
   - Measure throughput and latency

## Resources

### Books
- "Using MPI: Portable Parallel Programming with the Message-Passing Interface" by William Gropp, Ewing Lusk, and Anthony Skjellum
- "Using Advanced MPI: Modern Features of the Message-Passing Interface" by William Gropp, Torsten Hoefler, Rajeev Thakur, and Ewing Lusk
- "Parallel Programming with MPI" by Peter Pacheco
- "RDMA Aware Networks Programming User Manual" by Mellanox

### Online Resources
- [Open MPI Documentation](https://www.open-mpi.org/doc/)
- [MPICH Documentation](https://www.mpich.org/documentation/)
- [MPI Forum](https://www.mpi-forum.org/)
- [RDMA Consortium](http://www.rdmaconsortium.org/)
- [UCX Project](https://openucx.org/)
- [Mellanox RDMA Developer Documentation](https://docs.mellanox.com/display/RDMAmapsv60/)

### Video Courses
- "Parallel Programming with MPI" on Coursera
- "High Performance Computing" on edX
- "RDMA Programming" tutorials by various vendors

## Assessment Criteria

You should be able to:
- Implement efficient point-to-point and collective communication patterns
- Design parallel algorithms using appropriate MPI features
- Develop RDMA applications with proper memory management
- Analyze and optimize the performance of distributed applications
- Integrate GPU acceleration with MPI and RDMA
- Debug communication issues in distributed systems
- Select appropriate communication methods for specific problem domains

## Next Steps

After mastering MPI and RDMA programming, consider exploring:
- Partitioned Global Address Space (PGAS) models (UPC, Chapel)
- Task-based parallel programming models (Legion, HPX)
- Exascale computing challenges and solutions
- Specialized communication libraries for machine learning
- Fault tolerance in large-scale distributed systems
- Cloud-based HPC and distributed computing
