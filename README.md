# NASNet

## 0.Introduction
Neural networks have emerged as powerful and versatile models, demonstrating remarkable success across various challenging domains including computer vision, speech recognition, and natural language processing. However, designing effective neural network architectures remains a complex and challenging task, often requiring significant expertise and experimentation.
This challenge has sparked considerable research interest in automating the architecture design process, leading to the development of Neural Architecture Search (NAS). Notably, NAS-generated architectures have achieved state-of-the-art performance, frequently surpassing manually designed architectures across multiple benchmarks and tasks.
![](/imgs/overview.png)
Traditional NAS frameworks typically consist of three fundamental components: the search space, which defines the range of possible architectures; the search strategy, which determines how to explore this space; and the performance estimation strategy, which evaluates the effectiveness of candidate architectures. In one-shot NAS methods, the search strategy is uniquely integrated with the performance evaluation strategy, offering a more streamlined approach.


## 1.Search Space
The search space is a fundamental component of Neural Architecture Search (NAS). It represents a crucial balance between human design bias and search efficiency. A smaller search space with more predefined architectural decisions allows NAS algorithms to find high-performing architectures more quickly. Conversely, a larger search space with basic building blocks requires more computational time but offers the potential to discover innovative architectural designs.

### 1.1 NASNet search space
The NASNet search space pioneered the modern cell-based approach to architecture search. It draws inspiration from state-of-the-art human-designed CNN architectures, which typically feature repeated structural motifs. This search space explores two types of cells - normal cells and reduction cells - and stacks them to form the overall architecture while keeping the backbone (macro structure) fixed. The key distinction between these cells lies in the reduction cell's function: it halves the input height and width while doubling the number of filters to modify spatial resolution. To handle dimensional mismatches, 1×1 convolutions are strategically inserted where necessary.
 Examining the microstructure reveals how cells are formed: each cell comprises B blocks, with each block constructed through specific sequential steps 
![](/imgs/NASNet_Step.png)
During block construction, all unused hidden states generated within the convolutional cell are concatenated along the depth dimension to produce the final cell output(This is for NASNet-A architecture). The paper presents specific examples of discovered normal and reduction cells on CIFAR-10 dataset:
![](/imgs/NASNet_cell.png)

## 2.Search Strategy
 While the search space forms the foundation of NAS, the search strategy represents its most extensively studied component. These strategies generally fall into two main categories: black-box optimization techniques and one-shot techniques. Let's begin by examining some fundamental black-box optimization approaches.

### 2.1. Random Search
Random search serves as one of the most straightforward NAS baselines. It operates by randomly sampling architectures from the search space and fully training each one. The architecture that achieves the highest validation accuracy is selected as the final output. Despite its simplicity, numerous studies have demonstrated that random search can achieve surprisingly competitive performance.
 
### 2.2. RL based search
 Most reinforcement learning approaches model architecture design as a sequence of actions generated by a controller (typically an RNN). This controller is trained to maximize the expected validation accuracy of generated architectures.
![](/imgs/RL.png)
#### 2.2.1 Proximal policy optimization(PPO)
 PPO presents an alternative approach to training an RNN controller. It was originally proposed as an objective that achieves the data efficiency and reliable performance of Trust Region Policy Optimization (TRPO) while using only first-order optimizations.
TRPO maximizes a surrogate objective:

$ L^{CPI}(\theta)=\widehat{E}_{t}[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}]=\widehat{E}_{t}[r_t(\theta)\widehat{A}_ $

Without constraints, maximizing this objective would lead to excessively large policy updates. To address this issue, a CLIP objective was introduced:
$ L^{CLIP}(\theta)=\widehat{E}_t[min(r_t(\theta)\widehat{A}_t),clip(r_t(\theta),1-\epsilon,1+\epsilon)\widehat{A}_t] $
Due to clipping, excessively large policy updates will not change the objective value, effectively preventing large policy shifts. To ensure sufficient exploration, an entropy bonus is added. The final objective, with hyperparameter c, becomes:
$ L^{CLIP+S}(\theta)=\widehat{E}_t[L^{CLIP}(\theta)+cS[\pi_{\theta}](s_t)] $

### 2.3. Evolutionary and genetic algorithm based search
Evolutionary algorithms have gained significant popularity in architecture optimization due to their flexibility, conceptual simplicity, and competitive results.
![](/imgs/evolution.png)
 
#### 2.3.1 Regularized Evolution
 The Regularized Evolution algorithm represents a standard evolutionary method with a unique modification: it removes the longest-surviving architecture from the population in each step, regardless of its performance. This approach has demonstrated superior performance compared to both random search and RL methods, achieving state-of-the-art results on ImageNet upon its release in 2019.
![](/imgs/re.png)

### 2.4 One-shot methods
 The core concept behind one-shot methods is that every architecture produced by a NAS algorithm can be viewed as a subnetwork of a single "supernetwork." Once a supernet is trained, each architecture from the search space can be evaluated by inheriting its weights from the corresponding subnet within the supernet. This method allows training an exponential number of architectures for a linear computational cost. However, this approach relies on a crucial assumption: the ranking of architectures must remain relatively consistent with the ranking one would obtain from training them independently.

implementation OTW!!


## 3.Performance estimation strategy

### 3.1. Naive train and test
 The most straightforward way to evaluate an architecture is to train it on the training dataset and test it on the validation set. However, this approach is computationally expensive and significantly extends the NAS algorithm's running time.

### 3.2. Zero-cost proxies
 To accelerate the NAS algorithm, several approaches have been developed to perform rapid computations (such as a single forward and backward pass of a single minibatch of data) over a set of architectures. These methods aim to generate scores that highly correlate with final accuracies, forming the basic idea behind zero-cost proxies.

#### 3.2.1 Synflow
 Synflow represents the L1 path-norm of the network. According to our reference paper, Synflow outperformed five other zero-cost proxies. It can be expressed as:
synflow = i=0NLii

## 4.Implementation
example of the usage of my code is at “main.py” and example output of this file follows:
![](/imgs/result.png)
## 5.References

[NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.01578)

[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012)

[Neural Architecture Search: Insights from 1000 Papers](https://arxiv.org/pdf/2301.08727)

[Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548)

[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)


