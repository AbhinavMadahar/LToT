# Lateral Tree-of-Thoughts

**Author: Abhinav Madahar · अभिनव ਮਦਾਹਰ <abhinavmadahar@gmail.com>**

## Description

### Lateral Tree-of-Thoughts (LToT)

**LToT** is a reasoning architecture that extends vanilla *Tree-of-Thoughts* (ToT) by **explicitly modelling lateral thinking**.  In addition to the usual utility score *v*, every partial trace receives a **logical-consistency score** *c ∈ \[0, 1]*.  Branches are preserved if they are either high-utility (*mainline*) or low-utility / high-consistency (*lateral*), thereby converting surplus compute into principled diversity and mitigating premature pruning.

#### Why LToT?

* **Utility saturation:** frontier-scale LLMs quickly exhaust genuinely distinct high-utility continuations; further sampling yields near-duplicates.
* **Myopic pruning:** ToT discards logically sound but initially low-utility paths that might mature into correct solutions.
* LToT retains those paths as *lateral branches*, offering extra opportunities for error correction and creative problem-solving.

#### Core Algorithm (high level)

1. **Generate** *k* candidate continuations for each frontier trace.
2. **Score** each candidate with ⟨*v*, *c*⟩.
3. **Retain** candidates that are in the top-*p* quantile of *v* (mainline) **or** exceed a consistency threshold *cₘᵢₙ* (lateral).
4. **Search control:** alternate between exploiting mainlines and exploring laterals (policy still an open research question).
5. **Solution extraction:** return the best complete trace (or ensemble) found.

#### Current Research Directions

* **Consistency evaluation:** building reliable detectors for logically consistent, low-utility branches.
* **Exploration vs. exploitation:** formalising when to switch from mainline depth-first progress to lateral breadth-first rescue.

#### Roadmap

Near-term milestones are (i) a robust logical-consistency scorer and (ii) an adaptive exploration-exploitation policy; success would turn latent compute headroom into measurable reasoning gains.

## Guide to Reproducing the Results of (Madahar, 2025)

TODO
