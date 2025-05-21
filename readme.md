# Language Game-Based Fake News Propagation Model

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

## Implementation Details

### Key Components

1. **BeliefState**: Replaces the SIS categorization with a continuous belief state using Bayesian principles
2. **DOAgent** (Dynamic Opinion Agent): Implements agents using language games and active inference
3. **LanguageGameUtterance**: Represents communications between agents
4. **OfficialAgent**: Special agent that issues official announcements to counter fake news
5. **FPSWorldModel**: Runs the simulation using the language game framework

### Key Methods

- `_metropolis_hastings_update()`: Implements the M-H algorithm for belief updates
- `_minimize_free_energy()`: Applies active inference principles to minimize free energy
- `_calculate_expected_free_energy()`: Guides action selection in active inference
- `_run_language_game()`: Executes interactions between agents as language games
- `_process_utterance()`: Processes utterances using Bayesian inference

## Usage

### Basic Usage

```python
from language_game_model import run_model, analyze_results

# Run the model
model = run_model(
    num_agents=20,
    fake_news_topic="A study shows that drinking lemon water cures cancer",
    initial_infected=2,
    intervention_days=[5, 10, 15],
    steps=30
)

# Analyze results
results = analyze_results(model)

# Print summary
print(f"Final Statistics:")
for key, value in results["final_stats"].items():
    print(f"- {key}: {value}")
```

### Visualization

The model collects data compatible with visualization libraries like matplotlib:

```python
import matplotlib.pyplot as plt

# Get model data
model_data = model.datacollector.get_model_vars_dataframe()

# Plot belief states over time
model_data[["Susceptible", "Infected", "Recovered"]].plot()
plt.xlabel("Day")
plt.ylabel("Number of Agents")
plt.title("Fake News Propagation Dynamics")
plt.show()

# Plot average belief strength
model_data["AverageBeliefStrength"].plot()
plt.xlabel("Day")
plt.ylabel("Average Belief Strength")
plt.title("Evolution of Belief Strength")
plt.show()
```
![Image](https://github.com/user-attachments/assets/2a14fc44-03c1-4046-a045-f86d0624b3e5)
![Image](https://github.com/user-attachments/assets/13a50b5c-7384-41fc-a68a-511e0c7a8597)
![Image](https://github.com/user-attachments/assets/bf2fedf7-6bbf-45b7-85fc-41fc3fba7da7)

## Comparison to Original SIS Model

The language game approach offers several advantages over the original SIS model:

1. **Continuous Belief Representation**: Rather than discrete states, agents have continuous beliefs representing degrees of belief in fake news
2. **Sophisticated Belief Dynamics**: Uses Metropolis-Hastings algorithm and Bayesian inference for more realistic belief updates
3. **Active Decision-Making**: Agents actively select actions using active inference rather than passive state transitions
4. **Social Dynamics**: Better accounts for social influence, trust, and individual differences
5. **Principled Theoretical Foundation**: Based on well-established Bayesian and active inference frameworks

## Requirements

- Python 3.7+
- Mesa
- NumPy
- (Optional) OpenAI API for real LLM interactions in production

## Acknowledgments

This implementation builds upon the original work by Yuhan Liu et al. in the [FPS repository](https://github.com/LiuYuHan31/FPS), extending it with the language game approach.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
