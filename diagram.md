flowchart TD
    %% Top-Level Title
    A0(["DISTRIBUTED MLP TRAINING SYSTEM"])

    %% Master Process Block
    A1[/"MASTER PROCESS (Rank 0)"/]
    A1a[Data Loading & Preprocessing]
    A1b[Model Creation & DDP Wrapper]
    A1c[Results Aggregation & Visualization]
    A1 --> A1a
    A1 --> A1b
    A1 --> A1c

    %% Process Group Communication
    A2[/"PROCESS GROUP COMMUNICATION"/]
    A2a[Gloo Backend (CPU-based)]
    A2b[Gradient Sync (all_reduce)]
    A2c[Parameter Broadcasting (broadcast)]
    A2 --> A2a
    A2 --> A2b
    A2 --> A2c

    %% Worker Processes
    A3[/"WORKER PROCESSES"/]

    %% Worker 0
    W0["WORKER 0 (Rank 0)"]
    W0a[Data Partition 0<br/>(Batches 0,2,4...)]
    W0b[MLP Model Copy<br/>(Identical Parameters)]
    W0c[Forward Pass<br/>(Batch Processing)]
    W0d[Backward Pass<br/>(Gradient Computation)]
    W0 --> W0a --> W0b --> W0c --> W0d

    %% Worker 1
    W1["WORKER 1 (Rank 1)"]
    W1a[Data Partition 1<br/>(Batches 1,3,5...)]
    W1b[MLP Model Copy<br/>(Identical Parameters)]
    W1c[Forward Pass<br/>(Batch Processing)]
    W1d[Backward Pass<br/>(Gradient Computation)]
    W1 --> W1a --> W1b --> W1c --> W1d

    %% Execution Flow
    A4[/"EXECUTION FLOW"/]
    E1[1. Data Distribution<br/>(DistributedSampler)]
    E2[2. Forward Pass (Parallel)]
    E3[3. Loss Computation (Local)]
    E4[4. Backward Pass (Parallel)]
    E5[5. Gradient Sync (all_reduce)]
    E6[6. Parameter Update (Broadcast)]
    A4 --> E1 --> E2 --> E3 --> E4 --> E5 --> E6

    %% Connecting main flow
    A0 --> A1 --> A2 --> A3 --> A4
    A3 --> W0
    A3 --> W1
