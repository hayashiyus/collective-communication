## Language Game-Based Fake News Propagation Model

This repository contains an implementation of a language game-based approach to fake news propagation simulation. It replaces the epidemiological SIS (Susceptible-Infected-Susceptible) model used in the original [FPS repository](https://github.com/LiuYuHan31/FPS) with a more sophisticated Bayesian language game approach based on collective inference.

## Background

The original "From Skepticism to Acceptance: Simulating the Attitude Dynamics Toward Fake News" paper proposed a framework for simulating fake news propagation using LLM agents. However, its approach was limited by the use of an SIS epidemiological model rather than a realistic communication model.

This implementation addresses this limitation by replacing the SIS model with a language game-based approach that models communication and belief updates as a decentralized Bayesian inference process.

## Key Features

- **Bayesian Belief States**: Agents maintain continuous belief states representing their degrees of belief instead of discrete SIS categories
- **Metropolis-Hastings Updates**: Belief updates use the Metropolis-Hastings algorithm for more realistic exploration of belief spaces
- **Active Inference**: Agents select actions based on active inference principles, minimizing expected free energy
- **Language Games**: Communication is modeled as a language game where utterances influence beliefs through Bayesian inference
- **Social Network Effects**: Agent interactions are influenced by similarity and social network structure

## Theoretical Foundations

This implementation is based on three key papers:

1. **Generative Emergent Communication**: "Generative Emergent Communication: Large Language Model is a Collective World Model" ([arXiv:2501.00226](https://arxiv.org/abs/2501.00226))
   - Provides the theoretical framework for viewing language as emerging through decentralized Bayesian inference

2. **Metropolis-Hastings Captioning Game**: "Metropolis-Hastings Captioning Game: Knowledge Fusion of Vision Language Models via Decentralized Bayesian Inference" ([arXiv:2504.09620](https://arxiv.org/abs/2504.09620))
   - Offers the approach for using Metropolis-Hastings algorithm in language games

3. **Active Inference**: "Active Inference for Self-Organizing Multi-LLM Systems: A Bayesian Thermodynamic Approach to Adaptation" ([arXiv:2412.10425](https://arxiv.org/abs/2412.10425))
   - Provides the active inference framework used for agent action selection

## License

This project is licensed under the MIT License - see the LICENSE file for details.
