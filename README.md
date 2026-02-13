# Removing unwanted bias/information in deep neural networks using Information Theory

**Author:** Peter Bezak (e12220847@student.tuwien.ac.at)
**Advisors:** Univ. Ass. David Penz, Prof. Thomas Gärtner

---

## Problem Statement & Motivation

Multinomial Variational Autoencoders (MultVAE) are a variant of Variational Autoencoders that can be applied to collaborative filtering tasks. They map the high-dimensional sample space to a lower-dimensional latent space and use this new representation to sample and reconstruct the input.

MultVAE models used for collaborative filtering capture the patterns in the data; this includes underlying biases and demographic information. This can lead to strengthening existing echo chambers and possibly violating privacy. As such, it can be beneficial to remove certain sensitive information (e.g., gender, race, location) from latent space representations.

There are multiple approaches to debiasing MultVAE models. One example is training an adversarial network that predicts the protected attribute based on the latent space and then reverses the gradient. This mitigates the bias at the cost of a slight deterioration in performance.

Information theory offers an alternative path to debiasing in the form of mutual information (MI) estimation. It enables minimizing the MI between protected attributes and the latent space, which is a proven method in disentanglement tasks. There are multiple different MI estimation techniques; examples include CLUB, VUB, L1Out, and MMD.

### Research Questions

**RQ1 - Based on a literature search and analysis, which neural MI approximation methods are effective for MI minimization in disentanglement scenarios?**

**RQ2 - How effectively can the selected MI minimization techniques mitigate bias in MultVAE collaborative filtering tasks?**

**RQ3 - How does MI estimator-based minimization compare to using an adversarial network to remove protected information and what is the performance trade-off?**

---

## Methodology

As a first step, we conduct a literature search to survey existing methods for neural MI approximation and their applicability in the context of collaborative filtering (i.e., using MultVAE).

To compare different methods of bias mitigation using MI, we train multiple MultVAE networks with the aim of maximizing performance while minimizing the MI between the protected attribute and latent space. To gauge the effectiveness of bias removal, the balanced accuracy of an additional model that predicts the protected attribute based on the latent space is used.

To evaluate MI estimation as an alternative to adversarial training, we compare the performance of:

1. MultVAE networks debiased with adversarial training (baseline for bias mitigated performance).
2. MultVAE networks debiased with MI minimization.
3. Not debiased MultVAE networks (baseline for performance with bias).

Comparison is done on the commonly used **recall@10** and **NDCG@10** metrics. All of these models are trained with **MovieLens-1m**, **LFM-2b-DemoBias**, and **EB-NeRD** to provide robust comparisons across different datasets.

For training the MI minimization models, a disentanglement or domain adaptation approach is used. During training, there is a protected attribute extractor and a content extractor that both create their own latent space mapping. The protected attribute extractor is designed to extract all protected attribute information from the data; by then minimizing the mutual information between the two latent spaces, the algorithm can remove that bias from the content latent space.

---

## Related Work & State of the Art

For MI upper bound estimation and subsequent minimization, the CLUB paper presents multiple approaches and discusses how they can be implemented for MI minimization in information bottleneck (IB) and domain adaptation tasks. For IB specifically, the Deep Variational IB paper details a possible implementation, using VUB to minimize MI between the input and the latent representation.

In cases where the conditional probability is unknown, the CLUB paper presents a variational approach titled vCLUB, where the conditional probability is approximated by a neural network.

In the field of bias mitigation, a recent approach that uses adversarial training mitigates the bias by reversing the gradient of an adversarial network, trained to predict the protected variable, to remove the sensitive data with only a small performance decrease.

The FairMI paper applies the disentanglement approach with MI minimization using CLUB to collaborative filtering with Graph Convolutional Networks (GCN). It adds a novel learning objective, which is designed to minimize MI between the protected variable encoder and the information encoder, while simultaneously maximizing the MI between the information encoder and the user's interaction history conditioned on the protected variable encoder embedding.

We compare the adversarial approach of bias mitigation and multiple neural MI estimation approaches applied to collaborative filtering tasks using VAE models. We use a disentanglement approach similar to the FairMI paper but applied to a different model architecture and using multiple different MI estimation approaches. We expand on the literature in the field of bias mitigation by comparing the adversarial approach with using disentanglement in combination with different neural MI estimation approaches.

Would you like me to help you brainstorm potential diagrams or visual aids for the "Methodology" section to make the experimental setup clearer?