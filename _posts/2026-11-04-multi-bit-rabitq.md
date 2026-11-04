---
layout: post
title: "Multi-bit RaBitQ in ClickHouse: Exploring Vector Quantization, Algorithm Design, and Database Engineering"
date: 2026-11-04
permalink: /articles/multi-bit-rabitq.html
---


# 1. Introduction

### 1.1 Motivation

The rapid development of AI, particularly LLMs and multimodal models, has increased the demand for vector data. Text, images, and audio can be represented as high-dimensional embeddings, enabling semantic search and retrieval across different modalities [1, 2].

Vector quantization addresses these challenges through lossy compression. It encodes vectors using fewer bits, permanently discarding information in exchange for reduced storage requirements and lower computational costs. Methods such as Product Quantization (PQ) achieve this by quantizing vector subspaces independently [3].

The information loss introduced by quantization creates a fundamental trade-off between **Recall, Latency, and Storage**. More aggressive compression reduces storage and can lower query latency, but may degrade retrieval accuracy. Managing this trade-off is a central challenge in designing production vector search systems.

### 1.2 Research Overview and Contributions

Identifying optimization opportunities in ClickHouse's vector search implementation requires a systematic understanding of the performance characteristics of its existing quantization methods. To establish an empirical baseline, a unified benchmarking framework was developed to evaluate these methods under consistent experimental conditions.

The benchmark covers nine embedding datasets with different sources and dimensionalities, evaluating 14 quantizer configurations across 378 benchmark tasks. Each dataset uses up to 1,000 queries. Retrieval quality is measured using Recall@1, Recall@10, and Recall@100, while query latency is evaluated using median latency (p50). The benchmark scripts, dataset organization, experimental configurations, and result-generation procedures are maintained in a dedicated repository to support reproducibility.

**Benchmark Repository:** [clickhouse-vector-quantization-benchmarks](https://github.com/philip-jingxin/clickhouse-vector-quantization-benchmarks)
<img src="{{ '/assets/images/graph1_benchmark_coverage.png' | relative_url }}" alt="graph1_benchmark_coverage.png" width="596" style="max-width:100%;height:auto;">
[Figure 1: Benchmark Coverage Across Datasets and Quantizer Configurations]

The benchmark results reveal substantial differences in the performance characteristics of existing quantizers. [Figure 2] illustrates the distribution of Recall@10 and median query latency across the evaluated methods. Several low-latency configurations achieve moderate retrieval accuracy, including RaBitQ (0.6912 Recall@10 at 17.67 ms) and TurboQuant (0.6776 at 21.12 ms). Product Quantization improves Recall@10 to 0.8864, but with a higher median latency of 34.34 ms. INT8 achieves 0.9713 Recall@10 at 54.59 ms.

<img src="{{ '/assets/images/Existing_Quantizers_Recall10_Latency_linear.svg' | relative_url }}" alt="Existing_Quantizers_Recall10_Latency_linear.svg" width="636" style="max-width:100%;height:auto;">
[Figure 2: Recall@10 vs. Median Query Latency Across Existing Quantizers]

This performance distribution exposes a relatively underexplored region in the Recall–Latency space, particularly around Recall@10 values of 0.7–0.9 and median latency of 20–35 ms. Existing methods offer limited choices in this region, leaving an opportunity to achieve higher retrieval accuracy while maintaining competitive query latency.

This gap motivated further investigation into alternative quantization approaches, ultimately leading to the choice of **Multi-bit RaBitQ**, which extends the original binary RaBitQ representation to multiple bits per dimension. The implementation was evaluated at 2, 3, 4, and 6 bits across the same nine datasets, completing 108 benchmark tasks.

[Figure 3] overlays the Multi-bit RaBitQ results on the existing quantizer benchmark, showing that Multi-bit RaBitQ introduces competitive operating points within the previously underexplored region. At 3 bits, it achieves a median Recall@10 of 0.8849 at 25.15 ms, approaching the Recall of Product Quantization (0.8864) with approximately 26.8% lower median latency. At 4 bits, median Recall@10 increases to 0.9180 at 27.58 ms, representing a 3.16 percentage-point improvement in Recall@10 and a 19.7% reduction in median latency compared with Product Quantization. The 6-bit configuration further increases median Recall@10 to 0.9305 at 32.73 ms. These results demonstrate that Multi-bit RaBitQ achieves a more favorable Recall–Latency trade-off than several existing quantization methods, while its different bit-width configurations provide multiple operating points for balancing retrieval accuracy and query cost.

<img src="{{ '/assets/images/Recall10_vs_Median_Latency_log.png' | relative_url }}" alt="Recall10_vs_Median_Latency_log.png" width="651" style="max-width:100%;height:auto;">
[Figure 3: Recall@10 vs. Median Query Latency Across Existing Quantizers and Multi-bit RaBitQ]

### 1.3 Organization of the Article
The remainder of this article is organized as follows. Section 2 reviews existing quantization methods and their underlying principles. Section 3 describes the algorithmic exploration that motivated Multi-bit RaBitQ. Sections 4 and 5 present the algorithm design and its integration into ClickHouse. Section 6 analyzes the experimental results, examining how bit width, dataset characteristics, and algorithm design influence the observed Recall–Latency–Storage trade-offs. Sections 7 discuss the findings, limitations, future directions, and conclusions.

## 2. Background and Existing Quantization Methods

### 2.1 Vector Quantization

Given a database vector $$x \in \mathbb{R}^d$$ and a query vector $$q \in \mathbb{R}^d$$, vector quantization encodes the database vector into a compact representation:

$$
c = Q(x)
$$

where $$Q$$ denotes the quantization encoder and $$c$$ is the resulting code.

The distance between the query and the original vector is estimated from the quantized representation:

$$
\hat{d}(q,x) = F(q,c)
$$

where $$F$$ denotes the distance estimation function. The resulting estimation error is:

$$
\epsilon_d(q,x) = \hat{d}(q,x) - d(q,x)
$$

where $$d(q,x)$$ is the exact distance.

Different quantization methods employ different encoding functions $$Q$$ and distance estimation functions $$F$$, resulting in different storage costs, computational costs, and distance estimation accuracy.

### 2.2 Product Quantization (PQ)

Product Quantization (PQ) reduces codebook storage complexity by decomposing a high-dimensional vector into multiple low-dimensional subspaces [3].

Given a vector $$x$$ with dimension $$d$$, and a total encoding budget of $$B$$ bits, full-vector quantization requires $$2^B$$ centroids, each with $$d$$ dimensions. Its codebook storage is proportional to:

$$
S_{\mathrm{VQ}} = d \cdot 2^B
$$

PQ divides the vector into $$M$$ subspaces, each with $$d/M$$ dimensions. Each subspace uses $$b$$ bits, where:

$$
B = Mb
$$

Each subspace has an independent codebook containing $$2^b$$ centroids. The total codebook storage becomes:

$$
\begin{aligned}
S_{\mathrm{PQ}}
&= M \cdot 2^b \cdot \frac{d}{M} \\
&= d \cdot 2^b
\end{aligned}
$$

Thus, PQ reduces the codebook storage from:

$$
d \cdot 2^{Mb} \quad \text{to} \quad d \cdot 2^b
$$

while retaining the same number of possible combinations of subspace centroids:

$$
N_{\mathrm{PQ}} = (2^b)^M = 2^{Mb}
$$

During encoding, each subvector is assigned to its nearest centroid:

$$
i_j = \underset{k \in \{1,\ldots,2^b\}}{\arg\min}
\left\|x_j-p_{j,k}\right\|_2^2
$$

The reconstructed vector is formed by concatenating the selected centroids:

$$
\hat{x} = [p_{1,i_1},p_{2,i_2},\ldots,p_{M,i_M}]
$$

Since the subspaces are quantized independently, the total squared reconstruction error is:

$$
\left\|x-\hat{x}\right\|_2^2
=
\sum_{j=1}^{M}
\left\|x_j-p_{j,i_j}\right\|_2^2
$$

During query processing, PQ uses Asymmetric Distance Computation (ADC). The query remains unquantized, while the database vector is represented by its selected centroids:

$$
\hat{d}^2(q,x)
=
\sum_{j=1}^{M}
\left\|q_j-p_{j,i_j}\right\|_2^2
$$

Let $$e=x-\hat{x}$$ denote the reconstruction error. The squared distance estimation error is:

$$
\begin{aligned}
\hat{d}^2(q,x)-d^2(q,x)
&= \left\|q-\hat{x}\right\|_2^2-\left\|q-x\right\|_2^2 \\
&= 2(q-x)^T e+\left\|e\right\|_2^2
\end{aligned}
$$

This shows that distance estimation error depends on both the magnitude and direction of the reconstruction error.

PQ achieves lower codebook storage through independent subspace quantization. However, dependencies between different subspaces are not explicitly modeled, which can limit its representation accuracy.

### 2.3 TurboQuant

TurboQuant combines random rotation with coordinate-wise scalar quantization. Random rotation makes the coordinates approximately identically distributed and nearly independent in high-dimensional space, allowing each coordinate to be quantized independently [4].

Given a unit vector $$x \in \mathbb{R}^d$$, a random orthogonal transformation produces:

$$
\mathbf{y}=\mathbf{\Pi}\mathbf{x},
\qquad
\mathbf{\Pi}^{T}\mathbf{\Pi}=\mathbf{I}
$$

Each coordinate follows the marginal distribution:

$$
f_d(t)=
\frac{\Gamma(d/2)}
{\sqrt{\pi}\,\Gamma((d-1)/2)}
(1-t^2)^{(d-3)/2},
\qquad -1\leq t\leq 1
$$

For sufficiently large $$d$$, this distribution approaches a Gaussian distribution with variance $$1/d$$. TurboQuant uses a scalar codebook optimized for this distribution. Each coordinate is assigned to its nearest centroid:

$$
i_j=\arg\min_k |y_j-c_k|^2
$$

The reconstructed vector is:

$$
\hat{\mathbf{x}}=
\mathbf{\Pi}^{T}
[c_{i_1},c_{i_2},\ldots,c_{i_d}]^{T}
$$

For $$b$$ bits per coordinate, the MSE distortion for a unit vector is bounded by:

$$
\mathbb{E}\left[
\|\mathbf{x}-\hat{\mathbf{x}}\|_2^2
\right]
\leq
\frac{\sqrt{3}\pi}{2\cdot 4^b}
$$

For inner-product estimation, TurboQuant introduces a second stage using one additional bit to correct the residual left by MSE quantization. Let:

$$
\mathbf{r}=
\mathbf{x}-\hat{\mathbf{x}}_{\mathrm{MSE}}
$$

A one-bit Quantized Johnson–Lindenstrauss (QJL) transform encodes the residual:

$$
\mathbf{z}=
\operatorname{sign}(\mathbf{S}\mathbf{r}),
\qquad
S_{ij}\sim\mathcal{N}(0,1)
$$

Its reconstruction is:

$$
\hat{\mathbf{r}}=
\frac{\sqrt{\pi/2}}{d}\,
\|\mathbf{r}\|_2\,
\mathbf{S}^{T}\mathbf{z}
$$

The resulting inner-product estimator is unbiased:

$$
\mathbb{E}\left[
\langle\mathbf{q},
\hat{\mathbf{x}}_{\mathrm{MSE}}+\hat{\mathbf{r}}\rangle
\right]
=
\langle\mathbf{q},\mathbf{x}\rangle
$$

TurboQuant therefore provides two related quantization objectives: minimizing reconstruction error and reducing bias in inner-product estimation.

### 2.4 OPORP

OPORP (One Permutation + One Random Projection) is a randomized dimensionality reduction method. It applies a shared permutation and random sign projection, then aggregates the transformed coordinates into a smaller number of bins [5].

Given $$x\in\mathbb{R}^D$$, let $$\pi$$ be a random permutation and $$r_i$$ be independent Rademacher random variables:

$$
r_i\in\{-1,+1\},
\qquad
\Pr(r_i=1)=\Pr(r_i=-1)=\frac{1}{2}
$$

The permuted and signed coordinates are aggregated into $$k$$ bins:

$$
z_j(x)=
\sum_{i=1}^{D}
x_{\pi(i)}r_i I_{ij},
\qquad j=1,\ldots,k
$$

where $$I_{ij}=1$$ when coordinate $$i$$ is assigned to bin $$j$$, and $$0$$ otherwise. The resulting sketch is:

$$
\mathbf{z}(x)=
[z_1(x),z_2(x),\ldots,z_k(x)]^{T}
\in\mathbb{R}^{k}
$$

For two vectors $$u$$ and $$v$$, their cosine similarity can be estimated from the reduced representations:

$$
\hat{\rho}(u,v)=
\frac{
\langle\mathbf{z}(u),\mathbf{z}(v)\rangle
}{
\|\mathbf{z}(u)\|_2
\|\mathbf{z}(v)\|_2
}
$$

OPORP reduces the representation dimension from $$D$$ to $$k$$, lowering subsequent storage and computation costs. The approximation error depends on the projection and binning configuration; the original method provides a theoretical variance analysis for its similarity estimator.

### 2.5 RaBitQ

RaBitQ uses randomized binary quantization to represent normalized vectors with one bit per dimension. Its central idea is to construct a binary codebook and randomly rotate it to avoid favoring particular vector directions [6].

Given a data vector $$x_r$$, a centroid $$c$$ is used to normalize the vector:

$$
\mathbf{o}=
\frac{\mathbf{x}_r-\mathbf{c}}
{\|\mathbf{x}_r-\mathbf{c}\|_2}
$$

The binary codebook consists of all vertices of a normalized hypercube:

$$
\mathcal{C}=
\left\{
-\frac{1}{\sqrt{D}},
+\frac{1}{\sqrt{D}}
\right\}^{D}
$$

A random orthogonal transformation $$P$$ is applied to the codebook. The quantized vector is obtained by selecting the nearest codeword, which is equivalent to applying a sign function in the transformed coordinate system:

$$
\bar{\mathbf{o}}=
\frac{1}{\sqrt{D}}
\mathbf{P}
\operatorname{sign}(\mathbf{P}^{T}\mathbf{o})
$$

Given a normalized query vector $$q$$, RaBitQ estimates the inner product using:

$$
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
=
\frac{
\langle\bar{\mathbf{o}},\mathbf{q}\rangle
}{
\langle\bar{\mathbf{o}},\mathbf{o}\rangle
}
$$

The denominator can be precomputed during encoding. Under the randomized rotation model, this estimator is unbiased:

$$
\mathbb{E}_{P}\left[
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
\right]
=
\langle\mathbf{o},\mathbf{q}\rangle
$$

Its estimation error is bounded by $$O(1/\sqrt{D})$$ with high probability.

The original squared Euclidean distance can then be estimated by substituting the estimated inner product:

$$
\begin{aligned}
\hat{d}^{\,2}(\mathbf{x}_r,\mathbf{q}_r)
={}&
\|\mathbf{x}_r-\mathbf{c}\|_2^2
+\|\mathbf{q}_r-\mathbf{c}\|_2^2 \\
&-2\|\mathbf{x}_r-\mathbf{c}\|_2
\|\mathbf{q}_r-\mathbf{c}\|_2
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
\end{aligned}
$$

RaBitQ combines compact binary representations with an unbiased distance estimator and a probabilistic error bound.

### 2.6 IntN Quantization

IntN quantization maps each vector coordinate to a low-bit integer representation using a finite set of uniformly spaced quantization levels.

For a symmetric signed $$b$$-bit quantizer, define the maximum quantization level as:

$$
Q_{\max}=2^{b-1}-1
$$

Given a per-vector scale $$s$$, each coordinate is quantized as:

$$
z_i=
\operatorname{clip}
\left(
\operatorname{round}\left(\frac{x_i}{s}\right),
-Q_{\max},
Q_{\max}
\right)
$$

The reconstructed coordinate and quantization error are:

$$
\hat{x}_i=sz_i,
\qquad
e_i=x_i-\hat{x}_i
$$

When the quantization range is not exceeded, the rounding error satisfies:

$$
|e_i|\leq\frac{s}{2}
$$

Using the original query vector, the squared Euclidean distance is estimated from the reconstructed database vector:

$$
\hat{d}^{\,2}(q,x)=\|q-\hat{x}\|_2^2
$$

The corresponding distance estimation error is:

$$
\hat{d}^{\,2}(q,x)-d^2(q,x)
=
2(q-x)^T e+\|e\|_2^2
$$

For $$d$$-dimensional vectors, the encoded representation requires $$db$$ bits, excluding scale metadata. The quantization precision is controlled by the bit width and scale.

## 3. Exploring OPORP-based Quantization
###   3.1 Motivation and Experimental Design

The initial idea was to apply dimensionality reduction before vector quantization. By reducing the number of dimensions, the encoding budget could be allocated to fewer dimensions, allowing higher quantization precision.

Given an original vector $$\mathbf{x}\in\mathbb{R}^D$$, OPORP transforms it into a lower-dimensional representation:

$$  
\mathbf{z}=\operatorname{OPORP}(\mathbf{x}), \quad \mathbf{z}\in\mathbb{R}^k,\quad k<D  
$$

Under a fixed encoding budget of $$B$$ bits:

$$  
B=Db_0=kb_1  
$$

where $$b_0$$ and $$b_1$$ denote the average number of bits per dimension before and after dimensionality reduction, respectively. Thus:

$$  
b_1=\frac{D}{k}b_0  
$$

To evaluate this approach, OPORP was combined with two quantization methods: TurboQuant and IntN. The experiments examined different dimensionality reduction ratios and quantization bit widths, measuring their effects on Recall and Latency under a fixed Storage budget.

The same nine embedding datasets were used throughout the entire benchmark study, including the OPORP-based experiments and the evaluation of multi-bit RaBitQ. This ensures that all quantization methods are evaluated on a consistent set of datasets. The datasets are: gemini_001 (768 dimensions), gte_mul_ads (768), laion (768), openai_3_l (1536), openai_3_s (1536), wiki_cohere (1024), yfcc_clip (768), yfcc_nomic (768), and yfcc_siglip2 (1152).

###   3.2 OPORP + TurboQuant

OPORP was applied before TurboQuant to reduce the vector dimensionality. The OPORP compression ratio was varied from 1.5 to 3.0 to evaluate the effects of dimensionality reduction on quantization accuracy and query latency.

The following table summarizes the results across nine datasets. Recall values and p50 latency are reported as medians across datasets.

|OPORP Compression Ratio|Recall@1|Recall@10|Recall@100|p50 (ms)|
|---|---|---|---|---|
|1.5|0.6050|0.7035|0.7294|73.790|
|2.0|0.6190|0.6967|0.7284|69.276|
|2.5|0.5290|0.6586|0.6460|67.529|
|3.0|0.6280|0.7053|0.6932|66.616|

While median p50 latency decreased steadily by approximately 9.7% (from 73.79 ms to 66.62 ms), Recall@10 exhibited a non-monotonic pattern as the compression ratio increased. It first declined from 0.7035 to 0.6586, then rebounded to 0.7053 at a ratio of 3.0.

To examine dataset-level differences, the following table compares the mean cosine similarity of ground-truth top-10 neighbors with the Recall@10 change between compression ratios of 1.5 and 3.0.

|Dataset|Mean GT Top-10 Similarity|Recall@10 (1.5)|Recall@10 (3.0)|Change|
|---|---|---|---|---|
|yfcc_nomic|0.9273|0.5153|0.6689|+15.36 pp|
|gte_mul_ads|0.9220|0.6469|0.7626|+11.57 pp|
|yfcc_clip|0.8094|0.6467|0.6653|+1.86 pp|
|yfcc_siglip2|0.8013|0.7171|0.7290|+1.19 pp|
|laion|0.7202|0.7035|0.6689|-3.46 pp|
|gemini_001|0.7043|0.6223|0.5646|-5.77 pp|
|wiki_cohere|0.6469|0.7853|0.7307|-5.46 pp|
|openai_3_s|0.4976|0.7864|0.7103|-7.61 pp|
|openai_3_l|0.4785|0.7903|0.7053|-8.50 pp|

A clear relationship emerged between dataset similarity and the impact of dimensionality reduction. Datasets with higher ground-truth neighbor similarity generally showed improved Recall. For example, yfcc_nomic and gte_mul_ads achieved Recall@10 improvements of 15.36 and 11.57 percentage points, respectively. In contrast, openai_3_l and openai_3_s, which had substantially lower mean similarity scores, experienced decreases of 8.50 and 7.61 percentage points.

Across the nine datasets, the Spearman rank correlation between mean ground-truth top-10 similarity and Recall@10 change was 0.983. This indicates a strong association in the observed results, although it does not establish a causal relationship.

These results suggest that the effectiveness of combining OPORP with TurboQuant depends partly on the similarity characteristics of the original dataset. Dimensionality reduction affects different datasets differently, even when the same quantization method and compression settings are used.

###   3.3 OPORP + IntN

OPORP was combined with IntN under a fixed encoding budget. Four configurations were evaluated, pairing different dimensionality reduction ratios with IntN bit widths: 4x with 32 bits, 3x with 24 bits, 2x with 16 bits, and 1.5x with 12 bits. These configurations maintain the same nominal encoding budget by allocating fewer bits per dimension as more dimensions are retained.

The following table summarizes the median results across nine datasets.

|OPORP Compression Ratio|IntN Bits|Recall@1|Recall@10|Recall@100|p50 (ms)|
|---|---|---|---|---|---|
|4.0x|32|0.6300|0.6481|0.6356|94.252|
|3.0x|24|0.6900|0.7110|0.7016|118.604|
|2.0x|16|0.7750|0.7857|0.7796|172.849|
|1.5x|12|0.7860|0.8164|0.8166|223.912|

The per-dataset Recall@10 results are shown below.

|Dataset|4.0x / 32-bit|3.0x / 24-bit|2.0x / 16-bit|1.5x / 12-bit|Change (pp)|
|---|---|---|---|---|---|
|gemini_001|0.4919|0.5707|0.6818|0.7148|+22.29|
|gte_mul_ads|0.7597|0.8010|0.8492|0.8783|+11.86|
|laion|0.6097|0.6744|0.7567|0.7948|+18.51|
|openai_3_l|0.6481|0.7079|0.7857|0.8164|+16.83|
|openai_3_s|0.6551|0.7110|0.7868|0.8146|+15.95|
|wiki_cohere|0.7321|0.7329|0.8379|0.8233|+9.12|
|yfcc_clip|0.6179|0.6825|0.7696|0.7988|+18.09|
|yfcc_nomic|0.7127|0.7571|0.8248|0.8479|+13.52|
|yfcc_siglip2|0.6375|0.7511|0.7550|0.8413|+20.38|

Reducing the compression ratio from 4x to 1.5x improved Recall@10 across all nine datasets, with gains ranging from 9.12 to 22.29 percentage points. Eight datasets showed a monotonic increase across the four configurations. Median p50 latency increased from 94.252 ms to 223.912 ms over the same range, with increasing latency observed across all nine datasets.

Unlike OPORP + TurboQuant, whose Recall response varied substantially across datasets and exhibited a strong association with dataset similarity, OPORP + IntN showed a consistent Recall improvement as more dimensions were retained. Under a fixed storage budget, these results indicate that allocating the encoding budget across more dimensions with fewer bits per dimension can improve Recall, at the cost of increased query latency.

###   3.4 Experimental Findings and Design Implications

The experiments lead to two conclusions. First, the effectiveness of dimensionality reduction depends on the information-preservation characteristics of the quantizer. Increasing the bit width per dimension alone does not guarantee higher Recall. Second, the Recall–Latency trade-offs achieved by the OPORP-based combinations were insufficient to close the performance gap identified in the initial benchmark, as illustrated in the figure below.

<img src="{{ '/assets/images/Figure5_OPORP_Recall10_vs_Latency.svg' | relative_url }}" alt="Figure5_OPORP_Recall10_vs_Latency.svg" style="max-width:100%;height:auto;">
[Figure 4 OPORP based Quantization]

This motivated the exploration of multi-bit RaBitQ, which increases quantization precision while retaining the original vector dimensions, avoiding additional information loss from dimensionality reduction.

## 4. Multi-bit RaBitQ: Algorithm In Detail
### 4.1 Motivation

Multi-bit RaBitQ applies a shared random orthogonal transform and quantizes each dimension using uniform grid levels represented by $$B$$-bit integers. This increases quantization resolution while retaining the original dimensionality, providing a flexible trade-off between accuracy and storage. During search, inner products are estimated directly from quantized codes, transformed queries, and precomputed correction factors, avoiding vector reconstruction and enabling efficient distance estimation [7].

The Multi-bit RaBitQ paper reports strong accuracy across a range of compression rates. In its IVF-based ANN experiments, 4-bit, 5-bit, and 7-bit quantization achieve over 90%, 95%, and 99% recall, respectively, across all tested datasets, without accessing raw vectors for re-ranking. These results demonstrate the potential of Multi-bit RaBitQ to provide high retrieval accuracy while retaining the memory efficiency of quantized representations [7].

Multi-bit RaBitQ also offers two practical advantages for integration into ClickHouse:

- **SQL compatibility:** It can be exposed through the existing `Quantized` SQL syntax, keeping the user interface consistent with other quantization methods.
    
- **Implementation compatibility:** It can reuse and remain compatible with ClickHouse's existing RaBitQ implementation, allowing the new quantization method to fit into the established framework.

### 4.2 N-bit Quantization Model

Let $$\mathbf{o}_r,\mathbf{q}_r\in\mathbb{R}^D$$ denote a data vector and a query vector, respectively, and let $$\mathbf{c}$$ be a reference centroid. The normalized vectors are defined as:

$$
\mathbf{o}
=
\frac{\mathbf{o}_r-\mathbf{c}}
{\|\mathbf{o}_r-\mathbf{c}\|_2},
\qquad
\mathbf{q}
=
\frac{\mathbf{q}_r-\mathbf{c}}
{\|\mathbf{q}_r-\mathbf{c}\|_2}
$$

The squared Euclidean distance can be expressed using their inner product:

$$
\begin{aligned}
d^2(\mathbf{o}_r,\mathbf{q}_r)
={}&
\|\mathbf{o}_r-\mathbf{c}\|_2^2
+
\|\mathbf{q}_r-\mathbf{c}\|_2^2\\
&-
2\|\mathbf{o}_r-\mathbf{c}\|_2
\|\mathbf{q}_r-\mathbf{c}\|_2
\langle\mathbf{o},\mathbf{q}\rangle
\end{aligned}
$$

Therefore, distance estimation reduces to estimating the inner product between two unit vectors.

#### Codebook Construction

Let $$B$$ denote the number of bits per dimension. The scalar uniform grid is defined as:

$$
\mathcal{S}_B
=
\left\{
-\frac{2^B-1}{2}+u
\;\middle|\;
u=0,1,\ldots,2^B-1
\right\}
$$

The $$D$$-dimensional grid is its Cartesian product:

$$
\mathcal{G}_B=\mathcal{S}_B^D
$$

Each coordinate has $$2^B$$ possible values, resulting in $$2^{BD}$$ grid vectors.

Let $$P\in\mathbb{R}^{D\times D}$$ be a random orthogonal matrix:

$$
P^TP=PP^T=I
$$

The quantization codebook is constructed by normalizing each grid vector and applying the random orthogonal transformation:

$$
\mathcal{G}_r
=
\left\{
P\frac{\mathbf{y}}{\|\mathbf{y}\|_2}
\;\middle|\;
\mathbf{y}\in\mathcal{G}_B
\right\}
$$

When $$B=1$$, the scalar grid contains $$-\frac{1}{2}$$ and $$\frac{1}{2}$$. After normalization and rotation, the resulting codebook is identical to the original binary RaBitQ codebook [7].


### 4.3 Encoding Algorithm

#### Quantization Encoding

For a normalized data vector $$\mathbf{o}$$, the objective is to find its nearest codebook vector:

$$
\bar{\mathbf{o}}
=
\underset{\mathbf{v}\in\mathcal{G}_r}{\arg\min}
\|\mathbf{o}-\mathbf{v}\|_2^2
$$

Let $$\bar{\mathbf{y}}\in\mathcal{G}_B$$ be the corresponding grid vector:

$$
\bar{\mathbf{o}}
=
P\frac{\bar{\mathbf{y}}}{\|\bar{\mathbf{y}}\|_2}
$$

Since both vectors are unit vectors, minimizing the squared distance is equivalent to maximizing their inner product. Define the transformed data vector:

$$
\mathbf{o}'=P^{-1}\mathbf{o}
$$

The optimization becomes:

$$
\bar{\mathbf{y}}
=
\underset{\mathbf{y}\in\mathcal{G}_B}{\arg\max}
\frac{\langle\mathbf{y},\mathbf{o}'\rangle}
{\|\mathbf{y}\|_2}
$$

Exhaustively evaluating all $$2^{BD}$$ grid vectors is impractical. Lemma 3.1 establishes that there exists a scaling factor $$t>0$$ such that the optimal grid vector is also the nearest grid vector to the scaled vector $$t\mathbf{o}'$$:

$$
\bar{\mathbf{y}}
=
\underset{\mathbf{y}\in\mathcal{G}_B}{\arg\min}
\|t\mathbf{o}'-\mathbf{y}\|_2
$$

The nearest grid vector can be obtained through coordinate-wise rounding:

$$
\mathbf{y}(t)
=
\operatorname{Round}_{\mathcal{G}_B}
(t\mathbf{o}')
$$

Therefore, the search can be restricted to the critical scaling values at which the rounded vector changes.

For coordinate $$i$$, let $$s_i=\operatorname{sign}(\mathbf{o}'[i])$$. Since the optimal grid vector lies in the same orthant as $$\mathbf{o}'$$, the possible absolute coordinate values are:

$$
\frac{1}{2},\frac{3}{2},\ldots,
\frac{2^B-1}{2}
$$

The rounding result changes when the magnitude of the scaled coordinate crosses an integer boundary. The critical scaling values for coordinate $$i$$ are:

$$
t_{i,j}
=
\frac{j+1}{|\mathbf{o}'[i]|}
$$

where:

$$
j=0,1,\ldots,2^{B-1}-2
$$

for $$\mathbf{o}'[i]\neq 0$$. Coordinates with zero values do not generate critical scaling values.

The complete set of critical scaling values is:

$$
\mathcal{T}
=
\bigcup_{i:\mathbf{o}'[i]\neq 0}
\left\{
\frac{j+1}{|\mathbf{o}'[i]|}
\;\middle|\;
j=0,\ldots,2^{B-1}-2
\right\}
$$

The algorithm starts with the rounded vector at $$t=0$$, then enumerates the critical values in ascending order. Each critical value generates a new rounded grid vector. The best candidate is selected by maximizing:

$$
\frac{\langle\mathbf{y},\mathbf{o}'\rangle}
{\|\mathbf{y}\|_2}
$$

A min-heap is used to efficiently obtain the next smallest critical value. Since each update changes only one coordinate, the inner product and vector norm can be maintained incrementally.

After finding the optimal grid vector, it is shifted into an unsigned integer representation:

$$
\bar{\mathbf{y}}_u
=
\bar{\mathbf{y}}
+
\frac{2^B-1}{2}\mathbf{1}_D
$$

where $$\mathbf{1}_D$$ is the $$D$$-dimensional all-ones vector. Each coordinate is an unsigned integer:

$$
\bar{\mathbf{y}}_u[i]
\in
\{0,1,\ldots,2^B-1\}
$$

### 4.4 Query Processing and Distance Estimation

#### 4.4.1 Query Transformation

Given a query vector $$\mathbf{q}_r$$ and the reference centroid $$\mathbf{c}$$, the normalized query vector is:

$$
\mathbf{q}
=
\frac{\mathbf{q}_r-\mathbf{c}}
{\|\mathbf{q}_r-\mathbf{c}\|_2}
$$

The same random orthogonal transformation used during data encoding is applied to the query:

$$
\mathbf{q}'=P^{-1}\mathbf{q}
$$

The transformed query vector is reused across all database vectors. The query-dependent summation term can also be precomputed:

$$
\sum_{i=1}^{D}\mathbf{q}'[i]
$$

#### 4.4.2 Distance Estimation

Recall that the quantized database vector is represented as:

$$
\bar{\mathbf{o}}
=
P\frac{\bar{\mathbf{y}}}{\|\bar{\mathbf{y}}\|_2}
$$

Following the original RaBitQ estimator, the normalized inner product is estimated by:

$$
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
=
\frac{
\langle\bar{\mathbf{o}},\mathbf{q}\rangle
}{
\langle\bar{\mathbf{o}},\mathbf{o}\rangle
}
$$

Let $$\rho=\langle\bar{\mathbf{o}},\mathbf{o}\rangle$$ denote the precomputed correction factor. Using the unsigned integer representation:

$$
\bar{\mathbf{y}}_u
=
\bar{\mathbf{y}}
+
\frac{2^B-1}{2}\mathbf{1}_D
$$

the inner product can be computed directly from the quantized code:

$$
\boxed{
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
=
\frac{
\langle\bar{\mathbf{y}}_u,\mathbf{q}'\rangle
-
\frac{2^B-1}{2}
\sum_{i=1}^{D}\mathbf{q}'[i]
}{
\rho\|\bar{\mathbf{y}}\|_2
}
}
$$

The estimated squared Euclidean distance is obtained by substituting the estimated inner product:

$$
\boxed{
\begin{aligned}
\hat{d}^{\,2}(\mathbf{o}_r,\mathbf{q}_r)
={}&
\|\mathbf{o}_r-\mathbf{c}\|_2^2
+
\|\mathbf{q}_r-\mathbf{c}\|_2^2\\
&-
2\|\mathbf{o}_r-\mathbf{c}\|_2
\|\mathbf{q}_r-\mathbf{c}\|_2
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
\end{aligned}
}
$$

The database-dependent correction factor and vector norms can be precomputed during encoding, while the query transformation and summation term are computed once per query. The main per-vector computation is an inner product between the unsigned integer code and the transformed query.

#### 4.4.3 Error Estimation

Since the codebook consists of randomly rotated unit vectors and $$\bar{\mathbf{o}}$$ is the nearest codebook vector to $$\mathbf{o}$$, the estimator inherits the unbiasedness property of the original RaBitQ [7]:

$$
\mathbb{E}_P
\left[
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
\right]
=
\langle\mathbf{o},\mathbf{q}\rangle
$$

Its probabilistic error bound is:

$$
\boxed{
\begin{aligned}
\Pr_P\Bigg[
&
\left|
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
-
\langle\mathbf{o},\mathbf{q}\rangle
\right|\\
&\leq
\sqrt{
\frac{1-\rho^2}{\rho^2}
}
\cdot
\frac{\epsilon_0}{\sqrt{D-1}}
\Bigg]
\geq
1-\exp(-c_0\epsilon_0^2)
\end{aligned}
}
$$

where $$\rho=\langle\bar{\mathbf{o}},\mathbf{o}\rangle$$, $$c_0$$ is a positive constant, and $$\epsilon_0$$ controls the failure probability.

Since squared Euclidean distance is an affine function of the estimated inner product, its estimation error satisfies:

$$
\begin{aligned}
|\hat{d}^{\,2}-d^2|
={}&
2\|\mathbf{o}_r-\mathbf{c}\|_2
\|\mathbf{q}_r-\mathbf{c}\|_2\\
&\cdot
\left|
\widehat{\langle\mathbf{o},\mathbf{q}\rangle}
-
\langle\mathbf{o},\mathbf{q}\rangle
\right|
\end{aligned}
$$

Consequently, the squared distance estimator is also unbiased, with the following probabilistic error bound:

$$
\boxed{
\begin{aligned}
|\hat{d}^{\,2}-d^2|
\leq{}&
2\|\mathbf{o}_r-\mathbf{c}\|_2
\|\mathbf{q}_r-\mathbf{c}\|_2\\
&\cdot
\sqrt{\frac{1-\rho^2}{\rho^2}}
\frac{\epsilon_0}{\sqrt{D-1}}
\end{aligned}
}
$$

For normalized vectors, squared Euclidean distance is twice the cosine distance. Therefore, their estimation errors differ only by a factor of two, and the same analysis applies to cosine distance.

Theorem 3.2 further establishes the asymptotic optimality of Multi-bit RaBitQ. For an inner-product estimation error bound $$\epsilon$$ with failure probability at most $$\delta$$, under the condition:

$$
\frac{1}{\epsilon^2}
\log\frac{1}{\delta}
>
D
$$

the required number of bits per dimension satisfies:

$$
\boxed{
B
=
\Theta
\left(
\log
\left(
\frac{1}{D\epsilon^2}
\log\frac{1}{\delta}
\right)
\right)
}
$$

The total encoding size is $$BD$$ bits, establishing a logarithmic relationship between the required bit width and the inverse squared estimation error [7].

In addition to the theoretical guarantee, the paper provides an empirical relationship between estimation error, bit width, and dimensionality. Let $$\epsilon$$ denote the absolute error in estimating the inner product of two unit vectors. With more than 99.9% probability, the observed error satisfies:

$$
\boxed{
\epsilon
<
\frac{5.75}{2^B\sqrt{D}}
}
$$

The constant $$5.75$$ is obtained empirically and is not part of the theoretical guarantee [7]. This relationship indicates that the inner-product estimation error decreases exponentially with the bit width and inversely with the square root of the vector dimensionality.

### 4.5 Bit-width and Storage Analysis

For a $$D$$-dimensional vector with $$B$$ bits per dimension, the code storage is:

$$
S_{\mathrm{code}}=BD\text{ bits}
$$

The byte-aligned storage is:

$$
S_{\mathrm{code}}=
\left\lceil\frac{BD}{8}\right\rceil
\text{ bytes}
$$

The original Float32 representation requires:

$$
S_{\mathrm{FP32}}=32D\text{ bits}
$$

The compression ratio relative to Float32 is:

$$
R_{\mathrm{compression}}=\frac{32}{B}
$$

For $$N$$ database vectors, including auxiliary metadata:

$$
S_{\mathrm{total}}
=
N\left(
\left\lceil\frac{BD}{8}\right\rceil
+
S_{\mathrm{aux}}
\right)
$$

where $$S_{\mathrm{aux}}$$ represents the per-vector storage of correction factors and other metadata.

The following table compares different bit widths for 768-dimensional vectors.

| Bit width | Bits/vector | Bytes/vector | Compression vs. Float32 | Storage for 1M vectors |
| --------- | ----------: | -----------: | ----------------------: | ---------------------: |
| 1-bit     |         768 |           96 |                     32× |                  96 MB |
| 2-bit     |       1,536 |          192 |                     16× |                 192 MB |
| 3-bit     |       2,304 |          288 |                  10.67× |                 288 MB |
| 4-bit     |       3,072 |          384 |                      8× |                 384 MB |
| 6-bit     |       4,608 |          576 |                   5.33× |                 576 MB |
| Float32   |      24,576 |        3,072 |                      1× |               3.072 GB |

In conclusion, Multi-bit RaBitQ provides a linear storage trade-off controlled by bit width, allowing different configurations to balance compression efficiency and retrieval accuracy.

## 5. Engineering Implementation in ClickHouse

### 5.1 Clickhouse Quantization Framework Overview

ClickHouse organizes vector quantization into several components, with `VectorQuantizer` responsible for the actual quantization algorithms. The surrounding components handle codec configuration, column serialization, and vector search. [Figure 5] illustrates how these components interact and how `VectorQuantizer` participates in data encoding and query processing.

<img src="{{ '/assets/images/quantizer-arch.png' | relative_url }}" alt="quantizer-arch.png" style="max-width:100%;height:auto;">

**Figure 5. Relationships between ClickHouse Components**

The interaction between these components follows three stages:
1. **Table Creation:** `CompressionCodecQuantized` parses the quantization method and parameters, and calls `VectorQuantizer::validateParams()` to validate them. `ColumnsDescription` attaches `SerializationQuantizedVector` to the vector column.
    
2. **Data Insertion:** During MergeTree data part writing, `SerializationQuantizedVector` calls `VectorQuantizer::bytesPerVector()`, `createEncoder()`, and `encode()` to generate quantized codes. Original vectors and quantized codes are stored in separate subcolumns.
    
3. **Query Processing:** During vector search, `VectorQuantizer::prepareQuery()` prepares the query representation, and `distance()` calculates approximate distances from stored quantized codes. The results are used for candidate selection and full-precision re-scoring.

The main components and their responsibilities are summarized below.

| Component                      | Source file                                                     | Responsibility                                                                                                        |
| ------------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `CompressionCodecQuantized`    | `src/Compression/CompressionCodecQuantized.cpp`                 | Registers the `Quantized` codec, parses its parameters, and validates the configuration.                              |
| `ColumnsDescription`           | `src/Storages/ColumnsDescription.cpp`                           | Attaches `SerializationQuantizedVector`to the vector column.                                                          |
| `SerializationQuantizedVector` | `src/DataTypes/Serializations/SerializationQuantizedVector.cpp` | Manages vector serialization and invokes `VectorQuantizer` to generate quantized codes.                               |
| `VectorQuantizer`              | `src/Common/VectorQuantizer.cpp`                                | Implements quantization methods, encoder initialization, vector encoding, query preparation, and distance estimation. |

Within this architecture, `VectorQuantizer` provides the algorithm interfaces used by the surrounding components. [Figure 6] shows its internal organization.

<img src="{{ '/assets/images/vector-quantizer-arch.png' | relative_url }}" alt="vector-quantizer-arch.png" style="max-width:100%;height:auto;">
**Figure 6. Internal Structure of VectorQuantizer**

The internal interfaces are organized into three functional groups, following the quantizer registration mechanism:

- **Quantizer registration:** `QuantizationMethod` defines supported methods, and `methodToCodec()` maps method names to enum values for dispatch.
    
- **Configuration and storage:** `validateParams()` validates parameters, while `bytesPerVector()` determines the code size.
    
- **Encoding and query processing:** `createEncoder()` initializes the encoder, `encode()` quantizes vectors, `prepareQuery()` builds the query state, and `distance()` estimates distances from stored codes.

This organization allows each quantization method to provide its own encoding and distance estimation implementations while sharing the same external interfaces.

### 5.2 Registration and Configuration

Integrating Multi-bit RaBitQ into ClickHouse requires registering the quantization method and defining its supported parameters and storage size.

**Quantization Method Registration**

Multi-bit RaBitQ is added to the `QuantizationMethod` enum with the identifier `8`. The codec name `multi_bit_rabitq` is mapped to this enum value in `methodToCodec()`.

```cpp
enum class QuantizationMethod : UInt8
{
    MultiBitRaBitQ = 8,
};

if (method == "multi_bit_rabitq")
    return QuantizationMethod::MultiBitRaBitQ;
```

**Parameter Validation**

Multi-bit RaBitQ supports bit widths from 2 to 8 bits per dimension. The upper bound is set to 8 bits, at which the original paper reports 99% recall. In benchmarks, 6 bits already provides strong recall across the evaluated datasets.

```cpp
constexpr size_t MULTI_BIT_RABITQ_MIN_BITS = 2;
constexpr size_t MULTI_BIT_RABITQ_MAX_BITS = 8;
```

Also, `validateParams()` is modified to check that the dimensionality is a multiple of 8 and that the configured bit width is within the supported range.

**Storage Size**

`bytesPerVector()` calculates the encoded size for each vector. The implementation allocates one bit plane per bit, with a 4-byte correction factor appended:

```cpp
case QuantizationMethod::MultiBitRaBitQ:
    return (dimensions / 8) * bits + sizeof(float);
```

For a vector with dimensionality $$d$$ and bit width $$b$$, the storage size is:

$$  
\mathrm{code_size} = \frac{d}{8} \cdot b + 4  
$$
The extra 4-byte per vector is used to store the correction factor $$\alpha$$ used by the inner-product estimator:

$$  
\alpha = \frac{|\mathbf{x}|_2}{\langle \mathbf{x}, \hat{\mathbf{x}} \rangle}  
$$

where $$\mathbf{x}$$ is the projected data vector and $$\hat{\mathbf{x}}$$ is its quantized reconstruction. This factor is precomputed during encoding and multiplied by the approximate inner product during distance estimation to compensate for quantization distortion.

With this setup, Multi-bit RaBitQ can use the existing `VectorQuantizer` interfaces for encoding and query processing.

### 5.3 Data Encoding and Storage Layout

Multi-bit RaBitQ encoding is implemented through `VectorQuantizer::createEncoder()` and `VectorQuantizer::encode()`. [Figure 7] shows the encoding flow, including encoder initialization, scale search, bit-plane generation, and correction factor calculation.

<img src="{{ '/assets/images/encoding.png' | relative_url }}" alt="encoding.png" width="376" style="max-width:100%;height:auto;">
**Figure 7. Work flow of an Encoder **

| Step | Function                              | Responsibility                                                  |
| ---- | ------------------------------------- | --------------------------------------------------------------- |
| A    | `createEncoder()`                     | Initializes the random projection and reusable working buffers. |
| B    | `encode()` / `encodeMultiBitRaBitQ()` | Dispatches to and performs Multi-bit RaBitQ encoding.           |
| C    | `bestMultiBitRaBitQScale()`           | Searches for the quantization scale.                            |
| D    | `encodeMultiBitRaBitQ()`              | Generates quantization codes and packs them into bit planes.    |
| E    | `encodeMultiBitRaBitQ()`              | Calculates and stores the correction factor.                    |

**A. Encoder Initialization**

`createEncoder()` initializes the resources used during encoding.

```cpp
case QuantizationMethod::MultiBitRaBitQ:
{
    encoder->projection = generateRandomProjection(dimensions);
    encoder->work.resize(encoder->projection.size() / PROJECTION_ROUNDS);
    encoder->scale_events.reserve(dimensions);
    break;
}
```

`projection` defines the random rotation applied before quantization, as required by the original paper making the quantization error less dependent on the original coordinate distribution.

`scale_events` buffers the scale-transition events used by `bestMultiBitRaBitQScale()`. Each event records the scale at which a coordinate advances to its next quantization level, its normalized magnitude, and its current level:

```cpp
struct MultiBitRaBitQScaleEvent
{
    double scale;
    double magnitude;
    UInt32 level;
};
```

The event buffer is reused across vectors per each scale search to avoid repeated memory allocations.

**B. Vector Encoding**

`encodeMultiBitRaBitQ()` implements the encoding procedure described in Section 4.3.

**1. Random Projection and Normalization**

The input vector is first transformed using the random projection. The L2 norm of the projected vector is:

$$  
r=|\mathbf{o}'_{\mathrm{raw}}|_2  
=\sqrt{\sum_{i=1}^{D}\mathbf{o}'_{\mathrm{raw}}[i]^2}  
$$

The normalized projected vector is:

$$  
\mathbf{o}'=\frac{\mathbf{o}'_{\mathrm{raw}}}{r}  
$$

```cpp
applyRandomProjection(projection, x, dimensions, work);

double norm_sq = 0.0;

for (size_t i = 0; i < dimensions; ++i)
{
    const double value = static_cast<double>(work[i]);
    norm_sq += value * value;
}

const double norm = std::sqrt(norm_sq);
```

If the projected vector has zero norm, the encoder stores a zero correction factor and returns.

**2. Quantization Level Calculation**

The configured bit width $$(B)$$ is divided into one sign bit and $$(B-1)$$ magnitude bits. The maximum magnitude level is:

$$  
L_{\max}=2^{B-1}-1  
$$

The encoder searches for the optimal scaling factor $$(t^*)$$. For each coordinate, its normalized magnitude and quantization level are:

$$  
m_i=\frac{|\mathbf{o}'_{\mathrm{raw}}[i]|}{r}  
$$

$$  
\ell_i=  
\min\left(  
\left\lfloor t^*m_i\right\rfloor,  
L_{\max}  
\right)  
$$

```cpp
const size_t extra_bits = bits - 1;
const UInt32 sign_weight = UInt32{1} << extra_bits;
const UInt32 max_level = sign_weight - 1;

const double scale = bestMultiBitRaBitQScale(work, dimensions, norm, extra_bits, events);
```

**3. Signed-to-Unsigned Encoding**

The signed quantized coordinate is reconstructed from its sign and magnitude level:
$$
\operatorname{sign}(\mathbf{o}'_{\mathrm{raw}}[i])  
\left(\ell_i+\frac{1}{2}\right)  
$$

To store each coordinate as an unsigned (B)-bit integer, an offset is applied:
$$
\bar{y}[i]+\frac{2^B-1}{2}  
$$

Equivalently:

$$  
\bar{y}_u[i]=  
\begin{cases}  
2^{B-1}+\ell_i, & \mathbf{o}'_{\mathrm{raw}}[i]\geq 0\\  
2^{B-1}-1-\ell_i, & \mathbf{o}'_{\mathrm{raw}}[i]<0  
\end{cases}  
$$

The implementation calculates the normalized magnitude, quantization level, and unsigned code value:

```cpp
double full_code_dot = 0.0;

for (size_t i = 0; i < dimensions; ++i)
{
    const double value = static_cast<double>(work[i]);
    const double magnitude = std::abs(value) / norm;
    const UInt32 level = static_cast<UInt32>(std::min(static_cast<double>(max_level), std::floor(scale * magnitude)));
    const bool positive = value >= 0.0;
    const UInt32 code_value = positive ? (sign_weight | level) : (max_level - level);
```

Here, `code_value` corresponds to $$(\bar{y}_u[i])$$, the unsigned representation stored in the quantized code.

**4. Bit-plane Generation**

Each unsigned code value is represented using $$(B)$$ bits. The implementation stores the same bit position of all coordinates in a separate bit-plane:
$$
\bar{y}_u[i] \gg (B - 1 - p) \;\&\; 1
$$

where $$(p=0,\ldots,B-1)$$. Each plane occupies $$(D/8)$$ bytes, with coordinate $$(i)$$ stored at byte offset $$(i/8)$$ and bit offset $$(i\bmod 8)$$.

```cpp
    for (size_t plane = 0; plane < bits; ++plane)
    {
        const size_t bit_position = bits - 1 - plane;
        if (code_value & (UInt32{1} << bit_position))
            dst[plane * code_bytes + (i >> 3)] |= static_cast<char>(1u << (i & 7u));
    }
```

**5. Correction Factor Calculation**

The correction factor is derived from the inner product between the projected vector and its quantized reconstruction. Since:
$$
\operatorname{sign}(\mathbf{o}'_{\mathrm{raw}}[i])  
\left(\ell_i+\frac{1}{2}\right)  
$$

the inner product is accumulated as:

$$  
s=  
\sum_{i=1}^{D}  
\mathbf{o}'_{\mathrm{raw}}[i]\bar{y}[i]  
$$

The inverse correction factor stored for each vector is:

$$  
\alpha^{-1}=\frac{r}{s}  
$$

This is equivalent to the reciprocal of the correction factor defined in Section 4.4, since the normalized quantized vector is obtained by dividing $$(\bar{\mathbf{y}})$$ by its L2 norm. The implementation accumulates the inner product and stores the resulting factor:

```cpp
    const double centered_level = static_cast<double>(level) + 0.5;
    full_code_dot += value * (positive ? centered_level : -centered_level);
}

const float inv_factor = full_code_dot > 0.0 ? static_cast<float>(norm / full_code_dot) : 0.0f;
std::memcpy(dst + planes_bytes, &inv_factor, sizeof(inv_factor));
```

The final encoded representation consists of $$(B)$$ bit-planes followed by a 4-byte inverse correction factor.

**C. Scale Search**

`bestMultiBitRaBitQScale()` searches for the scale that maximizes the quantization objective. As the scale increases, coordinates advance through discrete quantization levels at specific transition points.

For coordinate $$(i)$$, with normalized magnitude $$(m_i)$$, the quantization level is:

$$  
\ell_i(t)=\min\left(\left\lfloor tm_i\right\rfloor,L_{\max}\right)  
$$

The transition to the next level occurs at:

$$  
t_i=\frac{\ell_i+1}{m_i}  
$$

**Initial Scale**

The search range is determined by the maximum coordinate magnitude and a bit-width-dependent starting ratio:

```cpp
static constexpr double tight_start[] = {0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81};

const UInt32 max_level = (UInt32{1} << extra_bits) - 1;
const double end_scale = (static_cast<double>(max_level) + 10.0) / max_magnitude;
const double start_scale = end_scale * tight_start[extra_bits];
```

The `tight_start` values are taken from the official RaBitQ-Library implementation. This  configuration narrows the search to the interval used by the reference implementation.

At the initial scale, the encoder calculates two accumulators:

$$  
N(t)=\sum_{i=1}^{D}m_i\left(\ell_i(t)+\frac{1}{2}\right)  
$$

$$  
Q(t)=\sum_{i=1}^{D}\left(\ell_i(t)+\frac{1}{2}\right)^2  
$$

The search maximizes the following objective:

$$  
S(t)=\frac{N(t)^2}{Q(t)}  
$$

```cpp
numerator += magnitude * (static_cast<double>(level) + 0.5);
denominator_sq += static_cast<double>(level) * (static_cast<double>(level) + 1.0);

if (magnitude > 0.0 && level < max_level)
    events.push_back({(static_cast<double>(level) + 1.0) / magnitude, magnitude, level});
```

**Heap-based Event Processing**

Each event records the next transition scale, coordinate magnitude, and current level. A min-heap orders events by transition scale:

```
const auto later_scale = [](const MultiBitRaBitQScaleEvent & lhs, const MultiBitRaBitQScaleEvent & rhs) { return lhs.scale > rhs.scale; };
std::make_heap(events.begin(), events.end(), later_scale);
```

When a coordinate advances from level $$(\ell_i)$$ to $$(\ell_i+1)$$, the accumulators are updated incrementally:

$$  
N \leftarrow N+m_i  
$$

$$  
Q \leftarrow Q+2(\ell_i+1)  
$$

The next transition for that coordinate is then inserted into the heap if its maximum level has not been reached.

```cpp
const UInt32 new_level = event.level + 1;
numerator += event.magnitude;
denominator_sq += 2.0 * (static_cast<double>(event.level) + 1.0);

if (new_level < max_level)
{
    events.push_back({(static_cast<double>(new_level) + 1.0) / event.magnitude, event.magnitude, new_level});
    std::push_heap(events.begin(), events.end(), later_scale);
}
```

Events sharing the same transition scale are processed together. After each group, the updated objective is evaluated. Since quantization levels remain unchanged between consecutive transition points, the midpoint of the interval is used as the candidate scale.

```cpp
const double score = numerator * numerator / denominator_sq;
next_scale = events.empty() ? end_scale : std::min(end_scale, events.front().scale);

if (score > best_score)
{
    best_score = score;
    best_scale = event_scale + (next_scale - event_scale) * 0.5;
}
```

The function returns the scale corresponding to the highest objective score. This scale is applied to all coordinates of the current vector during encoding.

### 5.4 Query Preparation and Distance Estimation

Multi-bit RaBitQ reuses the existing RaBitQ query representation and bit-sliced inner-product computation. The query is transformed once and reused across database vectors.

**A. Query Preparation**

`prepareQuery()` generates the same deterministic random projection used during data encoding, transforms the query, and builds the RaBitQ query representation.

```cpp
case QuantizationMethod::MultiBitRaBitQ:
{
    query->code_bytes = (dimensions / 8) * bits + sizeof(float);
    query->bits = bits;

    const std::vector<float> projection = generateRandomProjection(dimensions);
    std::vector<float> work(projection.size() / PROJECTION_ROUNDS);

    applyRandomProjection(projection, ref, dimensions, work.data());
    query->rabitq = buildRaBitQQuery(work.data(), dimensions);
    break;
}
```

`buildRaBitQQuery()` uniformly quantizes the transformed query into 4-bit values. Given the transformed query $$\mathbf{q}'$$, the quantization step is:

$$  
\Delta=\frac{q'_{\max}-q'_{\min}}{2^4-1}  
$$

Each coordinate is quantized as:

$$  
\tilde{q}_i=  
\operatorname{clip}  
\left(  
\operatorname{round}  
\left(  
\frac{q'[i]-q'_{\min}}{\Delta}  
\right),  
0,15  
\right)  
$$

The quantized query is stored as four bit planes. Its minimum value, quantization step, coordinate sum, and inverse L2 norm are also retained for inner-product estimation.

```cpp
constexpr int levels = (1 << RABITQ_QUERY_BITS) - 1;
const float range = q_max - q_min;
out.delta = (range > 0.0f) ? (range / static_cast<float>(levels)) : 0.0f;

const float inv_delta = (out.delta > 0.0f) ? (1.0f / out.delta) : 0.0f;

int q_tilde = static_cast<int>(std::lround((q[i] - q_min) * inv_delta));
q_tilde = std::clamp(q_tilde, 0, levels);

for (int j = 0; j < RABITQ_QUERY_BITS; ++j)
    if ((q_tilde >> j) & 1)
        out.planes[static_cast<size_t>(j) * out.code_bytes + (i >> 3)] |= static_cast<UInt8>(1u << (i & 7u));
```

**B. Bit-plane Inner Product**

For bit plane $$p$$, let $$b_{p,i}\in{0,1}$$ denote the stored bit at coordinate $$i$$. Its bipolar representation is:

$$  
s_{p,i}=2b_{p,i}-1  
$$

The existing `raBitQNumerator()` computes the approximate inner product between this sign representation and the transformed query:

$$  
d_p\approx\sum_{i=1}^{D}s_{p,i}q'[i]  
$$

The signed quantized coordinate can be reconstructed from all bit planes:
$$
\sum_{p=0}^{B-1}  
2^{B-1-p}  
\left(b_{p,i}-\frac{1}{2}\right)  
$$

Therefore, the inner product is accumulated across bit planes using their binary weights:

$$
N =
\frac{1}{2}
\sum_{p=0}^{B-1}
2^{B-1-p}d_p
$$

`multiBitRaBitQNumerator()` implements this weighted accumulation:

```cpp
float multiBitRaBitQNumerator(const RaBitQQuery & q, size_t bits, const char * code)
{
    float numerator = 0.0f;

    for (size_t plane = 0; plane < bits; ++plane)
    {
        const float plane_dot = raBitQNumerator(q, code + plane * q.code_bytes);
        const UInt32 weight = UInt32{1} << (bits - 1 - plane);
        numerator += 0.5f * static_cast<float>(weight) * plane_dot;
    }

    return numerator;
}
```

Each call to `raBitQNumerator()` uses the existing `AND + POPCNT` computation against the bit-sliced query.

**C. Distance Estimation**

During encoding, the correction factor is stored as:

$$  
\alpha=\frac{r}{s},  
\qquad  
s=\langle\mathbf{o}'_{\mathrm{raw}},\bar{\mathbf{y}}\rangle  
$$

The estimated cosine similarity is:
$$
\operatorname{clip}  
\left(  
\alpha  
\frac{  
\langle\bar{\mathbf{y}},\mathbf{q}'\rangle  
}{  
|\mathbf{q}'|_2  
},  
-1,1  
\right)  
$$

Using the precomputed inverse query norm and correction factor, the implementation calculates cosine distance:

$$
\hat{d}_{\cos}
=
1 -
\operatorname{clip}
\left(
N \cdot
\frac{1}{\|\mathbf{q}'\|_2}
\cdot \alpha,
-1,
1
\right)
$$

```cpp
float multiBitRaBitQDistance(const RaBitQQuery & q, size_t bits, const char * code)
{
    const size_t planes_bytes = bits * q.code_bytes;
    const float numerator = multiBitRaBitQNumerator(q, bits, code);

    float inv_factor = 0.0f;
    std::memcpy(&inv_factor, code + planes_bytes, sizeof(inv_factor));

    const float cosine = std::clamp(numerator * q.inv_qnorm * inv_factor, -1.0f, 1.0f);
    return 1.0f - cosine;
}
```

The correction factor is read immediately after the bit-plane region. The estimated cosine is clamped to $$[-1,1]$$ before conversion to cosine distance.

Multi-bit RaBitQ follows the cosine-distance path and is not listed as an L2-supported method in `supportsL2()`.
### 5.5 TMA Exploration and Evaluation

To investigate the microarchitectural behavior of Multi-bit RaBitQ during query execution, Top-down Microarchitecture Analysis (TMA)[8] was performed using `perf`. The profiling ran for approximately 965.5 seconds, with 964 sampling points, on a hybrid CPU with P-cores and E-cores.

**TMA Results**

The two groups of TMA metrics correspond to the two core types.

**P-core**

|TMA Metric|Mean|Median|Minimum|Maximum|
|---|---|---|---|---|
|Frontend Bound|9.43%|7.40%|4.9%|48.2%|
|Bad Speculation|2.19%|1.70%|0.5%|25.9%|
|Retiring|26.10%|23.60%|17.2%|64.3%|
|Backend Bound|62.29%|66.10%|13.3%|75.3%|

All 964 samples were valid. Backend Bound is the dominant metric, accounting for 62.29% on average. During the main sustained execution phase (300–900 seconds), it increases to approximately 69.29%.

**E-core**

|TMA Metric|Mean|Median|Minimum|Maximum|
|---|---|---|---|---|
|Frontend Bound|34.52%|37.90%|2.3%|55.2%|
|Bad Speculation|19.11%|21.70%|0.5%|44.7%|
|Retiring|20.07%|13.90%|4.5%|60.5%|
|Backend Bound|26.31%|24.30%|10.7%|77.4%|

Three anomalous samples were excluded, leaving 961 valid samples. Compared with the P-cores, the E-cores show higher Frontend Bound and Bad Speculation percentages, while Backend Bound is considerably lower.

**Function-Level Profiling and Optimization**

Function-level CPU sampling shows that vector data I/O, particularly reading and decompression, accounts for the largest costs.

| Function or Call Path                                                                              | Sampling Percentage |
| -------------------------------------------------------------------------------------------------- | ------------------- |
| `memcpy_ge64_baseline` ← `CompressionCodecQuantized::doDecompressData`                             | 15.56%              |
| `memcpy_ge64_baseline` ← `ReadBuffer::readStrict` / `CompressedReadBufferBase::readCompressedData` | 11.65%              |
| `raBitQNumerator()`                                                                                | 9.55%               |
| `VectorQuantizer::distance()` inclusive                                                            | 10.55%              |
| `VectorQuantizer::distance()` self                                                                 | 1.05%               |

Within distance calculation, the main computational cost comes from the existing `raBitQNumerator()`. A loop-fusion experiment was conducted to optimize this calculation, but its performance was comparable to the original implementation. 

**Conclusion**

The profiling results identify vector data I/O as a major source of query execution cost and show that P-core execution is predominantly Backend Bound. Although `raBitQNumerator()` is a visible computational hotspot, the loop-fusion experiment did not demonstrate a meaningful performance gain. The original implementation already provided sufficient performance, so no further change was adopted.

## 6. Benchmark and Performance Evaluation
### 6.1 Experimental Setup

#### 6.1.1 Hardware and Software Environment

All benchmarks were conducted on a machine equipped with an Intel Core i9-13900K processor and 61 GiB of memory. The processor has 24 physical cores and 32 logical CPUs, with a maximum frequency of 5.8 GHz. The system runs Linux with kernel version `7.0.0-38-generic`.

The benchmarks used ClickHouse 26.10.1.1. Existing quantizers were tested using the `master` build, while Multi-bit RaBitQ was tested using the feature build.

|Component|Configuration|
|---|---|
|CPU|Intel Core i9-13900K|
|Cores / Threads|24 / 32|
|Memory|61 GiB|
|OS Kernel|Linux 7.0.0-38-generic|
|ClickHouse|26.10.1.1|
|Benchmark Threads|1|

All queries used `max_threads = 1` and `enable_quantized_codec = 1`. Quantized codecs and Multi-bit RaBitQ used `vector_search_use_quantized_codes = 1`. Float32 and QBit used their respective query paths with this setting disabled.

#### 6.1.2 Datasets

The evaluation covers nine datasets with dimensions ranging from 768 to 1536. Each dataset contains 100,000 base vectors and 1,000 query vectors.

|Dataset|Dimensions|Base Vectors|Queries|
|---|---|---|---|
|gemini_001|768|100,000|1,000|
|gte_mul_ads|768|100,000|1,000|
|laion|768|100,000|1,000|
|openai_3_l|1536|100,000|1,000|
|openai_3_s|1536|100,000|1,000|
|wiki_cohere|1024|100,000|1,000|
|yfcc_clip|768|100,000|1,000|
|yfcc_nomic|768|100,000|1,000|
|yfcc_siglip2|1152|100,000|1,000|

Base vectors were imported from Parquet into MergeTree tables as `Array(Float32)`, with sequential IDs assigned in source-file order. 

Query vectors, ground truth, and query IDs were loaded from `fvecs` and `ivecs`. Each query's own base-vector ID was excluded from search and verified to be absent from the ground truth.

All benchmark queries use cosine distance, with ground truth generated using exact cosine similarity.

#### 6.1.3 Quantization Configurations

The baseline evaluation covers 14 configurations:

|Category|Configurations|
|---|---|
|Reference|Float32|
|Scalar quantization|INT8|
|Randomized quantization|TurboQuant|
|Binary quantization|RaBitQ|
|Prefix quantization|Prefix INT8, Prefix BF16|
|Product quantization|Product|
|QBit|1, 2, 3, 4, 6, 16, 32 bits|

Multi-bit RaBitQ was evaluated at 2, 3, 4, and 6 bits per dimension. The implementation supports 2–8 bits, while these four configurations were selected for the benchmark.

Each configuration was tested on all nine datasets at three retrieval cutoffs, $$K \in {1, 10, 100}$$.

| Benchmark           | Configurations | Datasets | Cutoffs | Tasks |
| ------------------- | -------------- | -------- | ------- | ----- |
| Existing quantizers | 14             | 9        | 3       | 378   |
| Multi-bit RaBitQ    | 4              | 9        | 3       | 108   |
|                     |                |          |         |       |

In total, 486 tasks were completed, covering 486,000 measured queries.

#### 6.1.4 Evaluation Metrics and Benchmark Procedure

**Recall**

Recall@K measures the fraction of ground-truth neighbors recovered in the returned top-K results:
$$
\frac{|G_K \cap R_K|}{K}  
$$

where $$G_K$$ and $$R_K$$ denote the ground-truth and returned top-K IDs, respectively. The reported Recall@K is the arithmetic mean across all 1,000 queries.

**Query Latency**

Latency is measured from HTTP request submission to response completion, including HTTP overhead and ClickHouse query execution. The reported statistics are `p50`, `p95`, `p99`, and `average` latency.

**Benchmark Procedure**

Each cutoff is benchmarked independently using its corresponding `LIMIT K`. Five warm-up queries are excluded, and each measured query runs once. Per-query results are saved to JSONL checkpoints, allowing interrupted runs to resume. Final reports aggregate Recall and latency by dataset, configuration, and cutoff.

### 6.2 Benchmark Results

The benchmark covers nine datasets, each containing 100,000 vectors. Results are grouped by quantizer and presented separately for each retrieval cutoff. All reported values are medians across the nine datasets. Latency is measured in milliseconds.

The complete benchmark results, including per-dataset measurements and per-query records, are available in the [reproducible benchmark repository](https://github.com/philip-jingxin/clickhouse-vector-quantization-benchmarks?utm_source=multi_bit_rabitq_article).

#### 6.2.1 Existing Quantizers

K = 1

|Quantizer|Recall@1|p50 (ms)|p95 (ms)|p99 (ms)|Average (ms)|
|---|---|---|---|---|---|
|Float32|1.0000|120.31|123.83|128.10|120.55|
|INT8|0.9610|43.84|46.04|50.33|44.06|
|Product Quantization|0.8370|23.28|25.60|31.04|23.53|
|TurboQuant|0.6240|10.71|15.11|18.21|11.15|
|RaBitQ|0.5810|7.21|10.24|13.40|7.43|
|Prefix INT8|0.6370|19.56|24.23|30.34|20.12|
|Prefix BF16|0.4830|10.85|15.03|18.52|11.13|
|QBit 1-bit|0.5800|6.89|11.96|14.56|7.63|
|QBit 2-bit|0.0000|30.03|31.00|34.82|30.14|
|QBit 3-bit|0.5800|43.18|44.15|49.41|43.23|
|QBit 4-bit|0.5800|55.56|56.55|60.21|55.46|
|QBit 6-bit|0.6420|82.10|84.02|85.80|82.32|
|QBit 16-bit|1.0000|243.07|248.57|251.49|244.30|
|QBit 32-bit|1.0000|430.19|438.26|442.28|428.91|

K = 10

|Quantizer|Recall@10|p50 (ms)|p95 (ms)|p99 (ms)|Average (ms)|
|---|---|---|---|---|---|
|Float32|1.0000|121.48|123.74|127.83|120.94|
|INT8|0.9713|54.59|59.36|62.01|54.69|
|Product Quantization|0.8864|34.34|39.30|43.43|34.34|
|TurboQuant|0.6776|21.11|25.86|27.89|21.29|
|RaBitQ|0.6912|17.67|22.17|24.64|17.75|
|Prefix INT8|0.6556|30.14|34.83|37.10|30.22|
|Prefix BF16|0.5067|21.20|26.05|29.68|21.36|
|QBit 1-bit|0.7115|6.91|10.52|15.04|7.30|
|QBit 2-bit|0.0001|30.20|31.54|37.80|30.41|
|QBit 3-bit|0.7122|42.89|44.21|50.01|43.21|
|QBit 4-bit|0.7122|55.71|56.59|60.78|55.57|
|QBit 6-bit|0.7664|82.10|84.20|88.36|82.40|
|QBit 16-bit|0.9991|247.34|249.84|254.27|246.19|
|QBit 32-bit|1.0000|427.48|437.47|441.34|427.73|

K = 100

|Quantizer|Recall@100|p50 (ms)|p95 (ms)|p99 (ms)|Average (ms)|
|---|---|---|---|---|---|
|Float32|1.0000|122.17|125.14|127.64|121.64|
|INT8|0.9827|93.49|98.83|101.01|92.27|
|Product Quantization|0.9108|72.18|78.79|80.93|71.49|
|TurboQuant|0.7659|57.36|65.15|67.33|57.32|
|RaBitQ|0.7308|54.71|61.29|63.09|54.17|
|Prefix INT8|0.6545|68.61|74.61|77.86|67.63|
|Prefix BF16|0.4918|57.66|65.38|67.11|57.71|
|QBit 1-bit|0.7217|7.08|9.20|12.80|7.29|
|QBit 2-bit|0.0010|30.10|31.04|34.86|30.18|
|QBit 3-bit|0.7222|43.63|44.64|49.82|43.73|
|QBit 4-bit|0.7222|55.66|56.89|58.63|55.54|
|QBit 6-bit|0.7746|82.19|84.28|87.39|82.17|
|QBit 16-bit|0.9993|244.23|249.21|252.37|245.11|
|QBit 32-bit|1.0000|429.17|439.23|444.55|428.93|

#### 6.2.2 Multi-bit RaBitQ

K = 1

|Bits per dimension|Recall@1|p50 (ms)|p95 (ms)|p99 (ms)|Average (ms)|
|---|---|---|---|---|---|
|2|0.7490|10.71|14.74|18.57|10.99|
|3|0.8410|13.93|18.25|21.75|14.43|
|4|0.8950|16.89|20.23|24.05|17.28|
|6|0.9070|22.69|26.37|30.86|23.08|

K = 10

|Bits per dimension|Recall@10|p50 (ms)|p95 (ms)|p99 (ms)|Average (ms)|
|---|---|---|---|---|---|
|2|0.8174|21.22|26.16|28.16|21.51|
|3|0.8849|25.15|30.08|32.11|25.27|
|4|0.9180|27.58|32.53|34.82|27.71|
|6|0.9305|32.73|39.96|43.55|32.98|

K = 100

|Bits per dimension|Recall@100|p50 (ms)|p95 (ms)|p99 (ms)|Average (ms)|
|---|---|---|---|---|---|
|2|0.8461|57.87|66.21|68.29|58.17|
|3|0.9069|62.91|69.38|71.14|62.06|
|4|0.9353|62.01|72.13|74.51|63.42|
|6|0.9477|73.53|80.20|82.42|72.62|

Across K = 1, 10, and 100, increasing the bit width from 2 to 6 improves recall from 0.7490 to 0.9070, 0.8174 to 0.9305, and 0.8461 to 0.9477, respectively. This comes with increased query latency, with p50 rising from 10.71 to 22.69 ms, 21.22 to 32.73 ms, and 57.87 to 73.53 ms at the three cutoffs. The 3-bit and 4-bit configurations provide intermediate trade-offs between recall and latency.

#### 6.2.3 Overall Comparison

The following table provides an overall comparison across all three retrieval cutoffs (K = 1, 10, and 100) and nine datasets. Recall is averaged over all 27 dataset–cutoff combinations for each quantizer. The average latency is calculated by taking the arithmetic mean of p50, p95, and p99 latency across the same 27 combinations. This provides a unified latency indicator for comparing the evaluated configurations.

|Quantizer|Average Recall|Average Latency (ms)|
|---|---|---|
|Float32|0.9998|135.83|
|INT8|0.9621|83.04|
|Product Quantization|0.8817|60.43|
|TurboQuant|0.6896|38.69|
|RaBitQ|0.6803|33.33|
|Prefix INT8|0.6140|50.29|
|Prefix BF16|0.4555|39.31|
|QBit 1-bit|0.6759|11.18|
|QBit 2-bit|0.0003|39.18|
|QBit 3-bit|0.6759|55.92|
|QBit 4-bit|0.6759|71.74|
|QBit 6-bit|0.7399|106.18|
|QBit 16-bit|0.9990|315.85|
|QBit 32-bit|0.9998|520.59|
|Multi-bit RaBitQ 2-bit|0.8171|38.91|
|Multi-bit RaBitQ 3-bit|0.8823|43.68|
|Multi-bit RaBitQ 4-bit|0.9131|47.01|
|Multi-bit RaBitQ 6-bit|0.9262|54.77|

<img src="{{ '/assets/images/overall_quantizer_recall_vs_latency_linear.png' | relative_url }}" alt="overall_quantizer_recall_vs_latency_linear.png" style="max-width:100%;height:auto;">
[Figure 8 Overall Quantizer Comparison: Recall vs. Latency in linear time]

## 7. Conclusion and Further Study

### 7.1 Conclusion

#### 7.1.1 Research Outcome

- OPORP experiments show that increasing the quantization bit width does not necessarily improve Recall when dimensionality reduction introduces additional information loss.
- Multi-bit RaBitQ demonstrates a stronger information-preservation capability than the currently evaluated quantizers, allowing higher Recall while maintaining competitive latency.
- The benchmark shows that increasing the bit width of Multi-bit RaBitQ consistently improves Recall, providing multiple operating points across the Recall–Latency trade-off.

#### 7.1.2 Engineering Outcome

- Multi-bit RaBitQ was integrated into the existing ClickHouse quantization framework, including codec registration, storage layout, encoding, query preparation, and distance estimation.
- Profiling and optimization experiments did not provide sufficient evidence that further low-level optimization would produce meaningful end-to-end performance gains.

### 7.2 Further Study

#### 7.2.1 Similarity Dependence in OPORP + TurboQuant

- Previous OPORP + TurboQuant experiments showed a strong dependence on nearest-neighbor similarity, particularly the similarity parameter \(\rho\).
- Further study could investigate the mathematical relationship between \(\rho\), dimensionality reduction error, quantization error, and Recall. This may clarify when dimensionality reduction is beneficial and when it causes excessive information loss.

#### 7.2.2 `tight_start` Parameter Selection

- The current `tight_start` values follow the empirical configuration used by RaBitQ-Library.
- The derivation and optimization procedure for these values is unclear.

#### 7.2.3 Information Preservation in Multi-bit RaBitQ

- The benchmark suggests that Multi-bit RaBitQ preserves information more effectively than several existing quantizers under comparable latency constraints.
- Further theoretical analysis could be developed to explain why Multi-bit RaBitQ preserves more information than other quantizers.

#### 7.2.4 Further Optimization Opportunities

- Current profiling does not identify an optimization with clear end-to-end benefit.
- Further exploration could examine vector loading, decompression, bit-plane distance computation, SIMD implementation, and query execution, under performance and practical engineering constraints.

#### 7.2.5 QBit 2-bit Recall Anomaly

- The benchmark shows unexpectedly low Recall for the QBit 2-bit configuration. Further investigation may identify the root cause.


---

## References

[1] Patrick Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020. [Paper](https://arxiv.org/abs/2005.11401)

[2] Alec Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. https://proceedings.mlr.press/v139/radford21a.html

[3] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011. https://inria.hal.science/inria-00514462v2

[4] Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. International Conference on Learning Representations (ICLR), 2026. [https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html)

[5] Ping Li and Xiaoyun Li. *OPORP: One Permutation + One Random Projection.* KDD, 2023. https://arxiv.org/abs/2302.03505

[6] Jianyang Gao and Cheng Long. *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search.* SIGMOD, 2024. https://doi.org/10.1145/3654970

[7] Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond Chi-Wing Wong. Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search. Proceedings of the ACM on Management of Data, 3(3), Article 202, 2025. [https://doi.org/10.1145/3725413](https://doi.org/10.1145/3725413)

[8] Ahmad Yasin. A Top-Down Method for Performance Analysis and Counters Architecture. IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2014, pp. 35–44. [https://doi.org/10.1109/ISPASS.2014.6844459](https://doi.org/10.1109/ISPASS.2014.6844459)