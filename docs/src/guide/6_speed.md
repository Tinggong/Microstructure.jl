# [Computation time] (@id guide-speed)

Table 1 summarizes the computation time for different tasks introduced in sections 3.2-3.3. Most analyses in this study were performed on a laptop equipped with an Apple M2 chip. On this system, the two-stage MCMC sampling required approximately 0.24 seconds per voxel using 8 threads. Training the neural network models for SMT and SANDI on the CPU took less than 10 seconds, indicating that GPU acceleration is not necessary for training small networks of this scale such as those demonstrated for SMT and SANDI. 

Considering that some compartment models, such as MTE models, will have higher dimensional inputs and outputs, leading to larger network models to train, we also tested training time for a network model with more units per hidden layer (150 vs. 48) and a larger training dataset (500,000 vs. 60,000 samples) than the one demonstrated for SANDI. Training this larger network model took approximately 6.5 minutes on the laptop and 18 minutes on a Linux workstation using the CPU. On the workstation, GPU acceleration significantly reduced training time for the larger model.

Table 1. Processing times.

|                    |  2-stage MCMC for estimating  |                   |                    |                      |
|                    |  axon diameter index; single  | Training for SMT  | Training for SANDI | Training larger model|  
|                    |  single-diffusion time data   |                   |                    |                      |
| ------------------ | ----------------------------- | ----------------- | ------------------ | -------------------- |
| Macbook Air laptop | 0.96 s/voxel (CPU; 1 thread)  | 2.8 s/model (CPU) |  8.1 s/model (CPU) | 6.5 min/model (CPU)  |  
| (Apple M2 16 GB)   | 0.24 s/voxel (CPU; 8 threads) |                   |                    |                      |
| ------------------ | ----------------------------- | ----------------- | ------------------ | -------------------- |
| Workstation with   |                               |                   |                    |                      |
| Intel Xeon w7-3455 | 1.17 s/voxel (CPU; 1 thread)  | 4.3 s/model (CPU) | 13.4 s/model (CPU) | 18.5 min/model (CPU) |
| processor and      |                               |                   |                    |                      | 
| NVIDIA RTX 6000 Ada| 0.15 s/voxel (CPU; 20 threads)|                   |                    |   6 min/model (GPU)  |
| Generation graphics|                               |                   |                    |                      | 
| card (48GB)        |                               |                   |                    |                      |


