"""
Language Game-Based Fake News Propagation Model

This module implements a language game-based approach to fake news propagation,
replacing the epidemiological SIS model with a Bayesian language game approach
based on Metropolis-Hastings inference.

The implementation is based on three key papers:
1. Generative Emergent Communication: LLM is a Collective World Model
2. Metropolis-Hastings Captioning Game (MHCG)
3. Active Inference for Self-Organizing Multi-LLM Systems

Author: Claude
Date: May 2025
"""

import numpy as np
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import openai
import json
import random
import math
from typing import List, Dict, Tuple, Optional, Any, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for Bayesian inference
TEMPERATURE = 0.7  # Temperature parameter for Metropolis-Hastings
BETA = 1.0  # Inverse temperature for active inference
BELIEF_THRESHOLD = 0.6  # Threshold for belief acceptance
FREE_ENERGY_WEIGHT = 0.8  # Weight for free energy minimization in active inference

class BeliefState:
    """
    Class to represent agent's belief state using Bayesian principles.
    Replaces the simple SIS classification with a continuous probability distribution.
    """
    def __init__(self, initial_belief: float = 0.0, certainty: float = 0.5):
        """
        Initialize belief state with:
        - initial_belief: probability of believing the fake news [0,1]
        - certainty: how certain the agent is about their belief [0,1]
        """
        self.belief = initial_belief  # P(fake news is true)
        self.certainty = certainty    # Precision of belief
        self.evidence_history = []    # Store past evidence to track changes
        self.free_energy = 0.0        # Free energy of current belief state

    def update_with_evidence(self, new_evidence: float, source_trustworthiness: float) -> None:
        """
        Update belief using Bayesian update rule with evidence weight based on source trust.
        
        Args:
            new_evidence: New evidence value [0,1] where 1 supports fake news being true
            source_trustworthiness: How trustworthy the source is [0,1]
        """
        # Store evidence for history tracking
        self.evidence_history.append((new_evidence, source_trustworthiness))
        
        # Prior (current belief)
        prior = self.belief
        
        # Likelihood P(evidence | hypothesis)
        # If evidence supports fake news (high value), likelihood is higher when belief is higher
        # Weight the evidence by source trustworthiness
        weighted_evidence = new_evidence * source_trustworthiness
        
        # Bayes update
        # P(H|E) ∝ P(E|H) * P(H)
        likelihood = weighted_evidence if prior > 0.5 else (1 - weighted_evidence)
        posterior_unnormalized = likelihood * prior
        
        # Normalize with marginal likelihood
        # P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)
        marginal = (weighted_evidence * prior) + ((1 - weighted_evidence) * (1 - prior))
        if marginal > 0:  # Avoid division by zero
            posterior = posterior_unnormalized / marginal
            
            # Update belief with smoothing based on certainty
            self.belief = (self.certainty * posterior) + ((1 - self.certainty) * prior)
            
            # Increase certainty slightly with each update (learning)
            self.certainty = min(0.95, self.certainty + 0.01)
            
            # Calculate free energy: F = -log P(evidence|belief) + KL[Q(belief)||P(belief)]
            prediction_error = -np.log(likelihood + 1e-10)  # Add small constant to avoid log(0)
            kl_divergence = self.belief * np.log((self.belief + 1e-10) / (prior + 1e-10)) + \
                           (1 - self.belief) * np.log((1 - self.belief + 1e-10) / (1 - prior + 1e-10))
            self.free_energy = prediction_error + kl_divergence

    def reset_certainty(self) -> None:
        """Reset certainty when agent encounters contradictory evidence."""
        self.certainty = max(0.3, self.certainty - 0.2)
    
    def get_belief_state(self) -> str:
        """
        Get a categorical belief state for compatibility with the original model.
        Returns: "susceptible", "infected", or "recovered"
        """
        if self.belief < 0.3:
            return "susceptible"
        elif self.belief > 0.7:
            return "infected"
        else:
            # Check if previously believed (was infected) but now uncertain
            was_infected = any(e > 0.7 for e, _ in self.evidence_history[:max(0, len(self.evidence_history)-5)])
            return "recovered" if was_infected else "susceptible"


class LanguageGameUtterance:
    """
    Class to represent utterances in the language game framework.
    An utterance contains content, belief strength, and metadata.
    """
    def __init__(self, 
                 content: str, 
                 belief_strength: float, 
                 source_agent_id: str,
                 is_official: bool = False):
        """
        Initialize an utterance in the language game.
        
        Args:
            content: The text content of the utterance
            belief_strength: How strongly the source believes [0,1]
            source_agent_id: ID of the source agent
            is_official: Whether this is from an official source
        """
        self.content = content
        self.belief_strength = belief_strength
        self.source_agent_id = source_agent_id
        self.is_official = is_official
        self.timestamp = 0  # To be set by the model
        
    def to_prompt_text(self) -> str:
        """Convert the utterance to text for prompting an LLM."""
        belief_text = "strongly believes" if self.belief_strength > 0.7 else \
                      "somewhat believes" if self.belief_strength > 0.3 else "doubts"
        
        source_text = "An official announcement" if self.is_official else f"Agent {self.source_agent_id}"
        
        return f"{source_text} {belief_text} the following: '{self.content}'"


class DOAgent(Agent):
    """
    Dynamic Opinion Agent (DOA) using language games and active inference.
    Replaces the SIS-based agent with one that participates in language games.
    """
    def __init__(self, 
                 unique_id: int, 
                 model: Model, 
                 name: str, 
                 age: int, 
                 education: str,
                 traits: List[str], 
                 initial_belief: float = 0.0,
                 openness: float = 0.5,
                 confirmatory_bias: float = 0.5) -> None:
        """
        Initialize a Dynamic Opinion Agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: The model the agent belongs to
            name: The agent's name
            age: The agent's age
            education: The agent's education level
            traits: List of personality traits
            initial_belief: Starting belief in fake news [0,1]
            openness: How open the agent is to new ideas [0,1]
            confirmatory_bias: Tendency to favor information confirming existing beliefs [0,1]
        """
        super().__init__(unique_id, model)
        
        # Agent persona
        self.name = name
        self.age = age
        self.education = education
        self.traits = traits
        self.openness = openness
        self.confirmatory_bias = confirmatory_bias
        
        # Belief state using Bayesian framework
        initial_certainty = random.uniform(0.3, 0.7)  # Randomize initial certainty
        self.belief_state = BeliefState(initial_belief, initial_certainty)
        
        # Memory components
        self.short_term_memory = []  # Recent interactions
        self.long_term_memory = []   # Persistent beliefs and important interactions
        
        # Language game components
        self.utterances_heard = []  # Utterances heard from other agents
        self.current_utterance = None  # Current utterance the agent is making
        self.communication_history = []  # History of all communications
        
        # Active inference components
        self.expected_free_energy = 0.0  # Expected free energy for action selection
        self.action_space = ["listen", "share_belief", "question", "reflect"]  # Possible actions
        self.action_priors = {a: 1.0/len(self.action_space) for a in self.action_space}  # Equal priors

    def get_persona_prompt(self) -> str:
        """Generate a prompt describing the agent's persona for the LLM."""
        traits_str = ", ".join(self.traits)
        return (f"You are {self.name}, {self.age} years old with {self.education} education. "
                f"Your personality traits include: {traits_str}. "
                f"You are {'very open to new ideas' if self.openness > 0.7 else 'somewhat open to new ideas' if self.openness > 0.3 else 'rather closed to new ideas'}. "
                f"You {'strongly tend to favor information that confirms your existing beliefs' if self.confirmatory_bias > 0.7 else 'somewhat tend to favor information that confirms your existing beliefs' if self.confirmatory_bias > 0.3 else 'try to evaluate information objectively'}.")

    def _generate_llm_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM based on a prompt.
        In a production system, this would call an actual LLM API.
        
        Args:
            prompt: The prompt for the LLM
            
        Returns:
            LLM-generated response as a string
        """
        try:
            # In production code, this would be an actual API call
            # Example with OpenAI:
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[{"role": "system", "content": self.get_persona_prompt()},
            #                {"role": "user", "content": prompt}],
            #     temperature=0.7,
            #     max_tokens=150
            # )
            # return response.choices[0].message["content"]
            
            # For demo purposes, just return a simulated response
            belief_level = self.belief_state.belief
            certainty = self.belief_state.certainty
            
            # Simulate response based on belief and certainty
            if belief_level > 0.7:
                return f"I believe this information is true. {random.choice(['It makes sense to me.', 'The evidence supports it.', 'I find it convincing.'])}"
            elif belief_level < 0.3:
                # Define choices outside of f-string to avoid backslash issues
                skeptical_phrases = ['It seems doubtful.', 'I need more evidence.', "This doesn't sound credible."]
                return f"I'm skeptical about this information. {random.choice(skeptical_phrases)}"
            else:
                # Define choices outside of f-string to avoid backslash issues
                uncertain_phrases = ['I need to consider it more.', 'I have mixed feelings.', 'It could be true, but I am uncertain.']
                return f"I'm not sure what to think about this. {random.choice(uncertain_phrases)}"
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I'm still considering this information."

    def listen(self, utterance: LanguageGameUtterance) -> None:
        """
        Agent listens to an utterance from another agent.
        
        Args:
            utterance: The utterance being received
        """
        # Store in short-term memory
        self.utterances_heard.append(utterance)
        self.short_term_memory.append(utterance)
        
        # Limit short-term memory size
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)
            
        # Process the utterance through active inference
        self._process_utterance(utterance)

    def _process_utterance(self, utterance: LanguageGameUtterance) -> None:
        """
        Process an utterance using Bayesian inference and active inference.
        
        Args:
            utterance: The utterance to process
        """
        # Calculate source trustworthiness
        # Official sources are given high trust
        if utterance.is_official:
            source_trust = 0.9
        else:
            # For other agents, trust is based on similarity and past interactions
            source_agent = self.model.get_agent_by_id(utterance.source_agent_id)
            similarity = self._calculate_agent_similarity(source_agent)
            source_trust = 0.3 + (0.6 * similarity)  # Scale to [0.3, 0.9]
        
        # Apply confirmatory bias
        # If the utterance aligns with current beliefs, trust is higher
        belief_alignment = 1 - abs(self.belief_state.belief - utterance.belief_strength)
        biased_trust = (source_trust * (1 - self.confirmatory_bias)) + \
                       (belief_alignment * self.confirmatory_bias)
        
        # Convert utterance to evidence
        evidence_value = utterance.belief_strength
        
        # Update belief using Bayesian update
        self.belief_state.update_with_evidence(evidence_value, biased_trust)
        
        # Store important utterances in long-term memory
        if biased_trust > 0.7 or abs(self.belief_state.belief - evidence_value) > 0.5:
            if len(self.long_term_memory) >= 10:
                self.long_term_memory.pop(0)
            self.long_term_memory.append(utterance)

    def _calculate_agent_similarity(self, other_agent: 'DOAgent') -> float:
        """
        Calculate similarity with another agent based on traits and beliefs.
        
        Args:
            other_agent: The other agent to compare with
            
        Returns:
            Similarity score [0,1]
        """
        # Calculate trait similarity
        common_traits = set(self.traits).intersection(set(other_agent.traits))
        trait_similarity = len(common_traits) / max(len(self.traits), len(other_agent.traits))
        
        # Calculate belief similarity
        belief_similarity = 1 - abs(self.belief_state.belief - other_agent.belief_state.belief)
        
        # Calculate demographic similarity (age, education)
        age_similarity = 1 - (abs(self.age - other_agent.age) / 100)  # Normalize by max age difference
        edu_similarity = 1 if self.education == other_agent.education else 0.5
        
        # Weighted average with higher weight on belief similarity
        return (0.2 * trait_similarity + 0.4 * belief_similarity + 
                0.2 * age_similarity + 0.2 * edu_similarity)

    def _metropolis_hastings_update(self, evidence: float, source_trust: float) -> bool:
        """
        Metropolis-Hastings algorithm for belief update, allowing for more 
        exploration of belief space than simple Bayesian updates.
        
        Args:
            evidence: New evidence value [0,1]
            source_trust: Source trustworthiness [0,1]
            
        Returns:
            Whether the proposed belief update was accepted
        """
        # Current belief state (prior)
        current_belief = self.belief_state.belief
        
        # Propose a new belief state (influenced by evidence)
        # Use trust to determine how much to move toward evidence
        proposed_shift = (evidence - current_belief) * source_trust
        
        # Add some noise for exploration
        noise = np.random.normal(0, 0.1 * (1 - self.belief_state.certainty))
        proposed_belief = current_belief + proposed_shift + noise
        
        # Ensure belief stays in [0,1] range
        proposed_belief = max(0.0, min(1.0, proposed_belief))
        
        # Calculate acceptance probability using Metropolis-Hastings
        # P(accept) = min(1, (P(proposed) * Q(current|proposed)) / (P(current) * Q(proposed|current)))
        
        # Likelihood of current and proposed beliefs given evidence
        current_likelihood = source_trust * (1 - abs(current_belief - evidence))
        proposed_likelihood = source_trust * (1 - abs(proposed_belief - evidence))
        
        # Prior probabilities (could incorporate other factors here)
        current_prior = 1.0  # Uniform prior for simplicity
        proposed_prior = 1.0
        
        # Calculate acceptance ratio
        # The proposal distribution Q is symmetric, so Q terms cancel out
        acceptance_ratio = (proposed_likelihood * proposed_prior) / \
                          (current_likelihood * current_prior)
        
        # Accept or reject based on ratio
        if acceptance_ratio >= 1 or random.random() < acceptance_ratio:
            # Update belief if accepted
            self.belief_state.belief = proposed_belief
            return True
        else:
            return False

    def _minimize_free_energy(self) -> None:
        """
        Apply active inference by minimizing free energy.
        This influences both belief updates and action selection.
        """
        # Calculate free energy for current belief state
        current_free_energy = self.belief_state.free_energy
        
        # Explore belief space to find lower free energy state
        for _ in range(5):  # Try a few iterations
            # Propose a small change to beliefs
            proposed_belief = self.belief_state.belief + np.random.normal(0, 0.05)
            proposed_belief = max(0.0, min(1.0, proposed_belief))
            
            # Calculate free energy for proposed belief
            
            # Prediction error component (simplified)
            evidence_values = [e for e, _ in self.belief_state.evidence_history[-5:]] if self.belief_state.evidence_history else [0.5]
            avg_evidence = sum(evidence_values) / len(evidence_values)
            prediction_error = abs(proposed_belief - avg_evidence)
            
            # Complexity/KL divergence component
            complexity = abs(proposed_belief - self.belief_state.belief)
            
            # Total free energy (weighted sum)
            proposed_free_energy = (0.7 * prediction_error) + (0.3 * complexity)
            
            # Accept if free energy is lower
            if proposed_free_energy < current_free_energy:
                self.belief_state.belief = proposed_belief
                self.belief_state.free_energy = proposed_free_energy
                current_free_energy = proposed_free_energy

    def _calculate_expected_free_energy(self, action: str) -> float:
        """
        Calculate expected free energy for an action.
        This guides action selection in active inference.
        
        Args:
            action: The action to evaluate
            
        Returns:
            Expected free energy value
        """
        # Base components of expected free energy:
        # 1. Epistemic value (information gain)
        # 2. Pragmatic value (goal-directed value)
        
        if action == "listen":
            # Listening has high epistemic value when uncertainty is high
            epistemic_value = 1 - self.belief_state.certainty
            pragmatic_value = 0.5  # Neutral pragmatic value
            
        elif action == "share_belief":
            # Sharing has high pragmatic value when certainty is high
            epistemic_value = 0.3  # Modest information gain
            pragmatic_value = self.belief_state.certainty
            
        elif action == "question":
            # Questioning has high epistemic value when belief is moderate
            mid_belief = abs(self.belief_state.belief - 0.5)
            epistemic_value = 0.7 * (1 - mid_belief)  # Higher for beliefs around 0.5
            pragmatic_value = 0.5
            
        elif action == "reflect":
            # Reflection has high epistemic value with conflicting evidence
            conflicting_evidence = False
            if len(self.belief_state.evidence_history) >= 3:
                recent_values = [e for e, _ in self.belief_state.evidence_history[-3:]]
                variance = np.var(recent_values) if recent_values else 0
                conflicting_evidence = variance > 0.2
                
            epistemic_value = 0.8 if conflicting_evidence else 0.4
            pragmatic_value = 0.6  # Moderately valuable for goal pursuit
            
        else:
            return 0.0  # Unknown action
        
        # Calculate expected free energy
        # Lower is better in active inference
        expected_free_energy = -(epistemic_value + pragmatic_value)
        
        # Incorporate action priors
        expected_free_energy += -np.log(self.action_priors.get(action, 0.1) + 1e-10)
        
        return expected_free_energy

    def select_action(self) -> str:
        """
        Select an action using active inference principles.
        
        Returns:
            Selected action
        """
        # Calculate expected free energy for each action
        action_values = {
            action: self._calculate_expected_free_energy(action)
            for action in self.action_space
        }
        
        # Convert to probabilities using softmax
        values = np.array(list(action_values.values()))
        probabilities = np.exp(-BETA * values) / np.sum(np.exp(-BETA * values))
        
        # Select action based on probabilities
        action_idx = np.random.choice(len(self.action_space), p=probabilities)
        selected_action = self.action_space[action_idx]
        
        # Update action priors (learning)
        for i, action in enumerate(self.action_space):
            if i == action_idx:
                self.action_priors[action] = 0.9 * self.action_priors[action] + 0.1
            else:
                self.action_priors[action] = 0.9 * self.action_priors[action]
                
        return selected_action

    def generate_utterance(self) -> LanguageGameUtterance:
        """
        Generate an utterance to share with other agents.
        
        Returns:
            A new utterance
        """
        # Create utterance content based on current beliefs and topic
        prompt = (
            f"Based on your current understanding of the topic '{self.model.fake_news_topic}', "
            f"express your opinion in a short statement. "
            f"Current belief level: {'high' if self.belief_state.belief > 0.7 else 'low' if self.belief_state.belief < 0.3 else 'moderate'}. "
            f"Certainty: {'high' if self.belief_state.certainty > 0.7 else 'low' if self.belief_state.certainty < 0.3 else 'moderate'}."
        )
        
        # Get content from LLM
        content = self._generate_llm_response(prompt)
        
        # Create and return utterance
        utterance = LanguageGameUtterance(
            content=content,
            belief_strength=self.belief_state.belief,
            source_agent_id=self.unique_id,
            is_official=False
        )
        
        self.current_utterance = utterance
        return utterance

    def reflect(self) -> None:
        """
        Agent reflects on recent interactions and updates beliefs.
        Uses Metropolis-Hastings and active inference.
        """
        # Skip reflection if no recent interactions
        if not self.utterances_heard:
            return
            
        # Create reflection prompt based on recent interactions
        recent_utterances = self.utterances_heard[-5:]
        utterance_texts = [u.to_prompt_text() for u in recent_utterances]
        
        prompt = (
            f"Based on the following recent interactions regarding '{self.model.fake_news_topic}', "
            f"reflect on your current beliefs. Do you find this information credible? Why or why not?\n\n"
            + "\n".join(utterance_texts)
        )
        
        # Get reflection from LLM
        reflection = self._generate_llm_response(prompt)
        
        # Extract belief signals from reflection (in production, would use NLP or LLM)
        # For demo, we'll use a simple approach
        positive_signals = ["credible", "believe", "true", "convinced", "evidence supports"]
        negative_signals = ["skeptical", "false", "unconvinced", "doubtful", "need more evidence"]
        
        # Count positive and negative signals
        positive_count = sum(1 for signal in positive_signals if signal.lower() in reflection.lower())
        negative_count = sum(1 for signal in negative_signals if signal.lower() in reflection.lower())
        
        # Convert to belief signal
        if positive_count > negative_count:
            reflection_belief = 0.6 + (0.4 * (positive_count / (positive_count + negative_count + 1)))
        elif negative_count > positive_count:
            reflection_belief = 0.4 * (1 - (negative_count / (positive_count + negative_count + 1)))
        else:
            reflection_belief = 0.5
            
        # Apply Metropolis-Hastings update
        self._metropolis_hastings_update(reflection_belief, 0.8)  # High trust in own reflection
        
        # Minimize free energy
        self._minimize_free_energy()
        
        # Update certainty based on reflection
        self.belief_state.certainty = min(0.95, self.belief_state.certainty + 0.05)
        
        # Clear heard utterances after reflection
        self.utterances_heard = []

    def step(self) -> None:
        """Agent's step function in the simulation."""
        # Select action using active inference
        action = self.select_action()
        
        # Execute action
        if action == "listen":
            # Agent already listens in interaction phase
            pass
        elif action == "share_belief":
            # Create utterance to share
            self.generate_utterance()
        elif action == "question":
            # Question most recent utterance (simulated by increasing uncertainty)
            if self.utterances_heard:
                # Target the most recent utterance with highest impact
                latest = self.utterances_heard[-1]
                source_agent = self.model.get_agent_by_id(latest.source_agent_id)
                if source_agent:
                    # Make the source agent less certain
                    source_agent.belief_state.certainty = max(0.2, source_agent.belief_state.certainty - 0.1)
        elif action == "reflect":
            self.reflect()


class OfficialAgent(DOAgent):
    """Official agent that issues announcements to counter fake news."""
    def __init__(self, 
                 unique_id: int, 
                 model: Model, 
                 intervention_schedule: List[int]) -> None:
        """
        Initialize an official agent.
        
        Args:
            unique_id: Unique identifier
            model: The model
            intervention_schedule: List of days when interventions occur
        """
        super().__init__(
            unique_id, 
            model, 
            name="Official Source", 
            age=0,  # Not relevant for official
            education="Expert",
            traits=["authoritative", "knowledgeable", "trustworthy"],
            initial_belief=0.0  # Officials know the news is fake
        )
        
        self.intervention_schedule = intervention_schedule
        self.is_official = True

    def step(self) -> None:
        """Official agent step function."""
        # Check if today is an intervention day
        if self.model.schedule.time in self.intervention_schedule:
            self.intervene()

    def intervene(self) -> None:
        """Issue an official intervention to counter fake news."""
        content = (
            f"Official Announcement regarding '{self.model.fake_news_topic}': "
            f"After careful investigation, we have determined that this information is false. "
            f"Please be cautious about sharing unverified claims."
        )
        
        utterance = LanguageGameUtterance(
            content=content,
            belief_strength=0.0,  # Strong disbelief in fake news
            source_agent_id=self.unique_id,
            is_official=True
        )
        
        # Send announcement to all agents
        for agent in self.model.schedule.agents:
            if isinstance(agent, DOAgent) and agent.unique_id != self.unique_id:
                agent.listen(utterance)


class FPSWorldModel(mesa.Model):
    """
    Model class for the Fake news Propagation Simulation using language games.
    Replaces the SIS model with a Bayesian language game approach.
    """
    def __init__(self, 
                 N: int = 20, 
                 fake_news_topic: str = "UFO sighting confirmed by government",
                 initial_infected_count: int = 1,
                 intervention_days: List[int] = None) -> None:
        """
        Initialize the fake news propagation model.
        
        Args:
            N: Number of agents
            fake_news_topic: The fake news topic being propagated
            initial_infected_count: Number of initially infected agents
            intervention_days: Days when official intervention occurs
        """
        # Initialize parent class
        super().__init__()
        
        self.num_agents = N
        self.fake_news_topic = fake_news_topic
        self.initial_infected_count = initial_infected_count
        self.intervention_days = intervention_days or [5, 10, 15]
        
        # Set up the model
        self.schedule = RandomActivation(self)
        
        # Create regular agents
        self._create_agents()
        
        # Create official agent for interventions
        official_agent = OfficialAgent(self.num_agents + 1, self, self.intervention_days)
        self.schedule.add(official_agent)
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: self._count_susceptible(m),
                "Infected": lambda m: self._count_infected(m),
                "Recovered": lambda m: self._count_recovered(m),
                "AverageBeliefStrength": lambda m: self._average_belief_strength(m),
                "BeliefVariance": lambda m: self._belief_variance(m)
            },
            agent_reporters={
                "Belief": lambda a: a.belief_state.belief if isinstance(a, DOAgent) else None,
                "Certainty": lambda a: a.belief_state.certainty if isinstance(a, DOAgent) else None,
                "State": lambda a: a.belief_state.get_belief_state() if isinstance(a, DOAgent) else None
            }
        )
        
        # Collect initial data
        self.datacollector.collect(self)
        
        # Initialize interaction network (who interacts with whom)
        self.interaction_network = self._initialize_interaction_network()
        
        # Track simulation variables
        self.running = True
        self.day = 0

    def _create_agents(self) -> None:
        """Create and initialize agents for the simulation."""
        # Generate diverse personas
        personas = self._generate_diverse_personas(self.num_agents)
        
        # Create agents with those personas
        for i in range(self.num_agents):
            persona = personas[i]
            
            # Determine initial belief (mostly skeptical, a few believers)
            initial_belief = random.uniform(0.7, 0.9) if i < self.initial_infected_count else random.uniform(0.0, 0.3)
            
            # Create agent
            agent = DOAgent(
                i, 
                self,
                name=persona["name"],
                age=persona["age"],
                education=persona["education"],
                traits=persona["traits"],
                initial_belief=initial_belief,
                openness=persona["openness"],
                confirmatory_bias=persona["confirmatory_bias"]
            )
            
            self.schedule.add(agent)

    def _generate_diverse_personas(self, count: int) -> List[Dict]:
        """
        Generate diverse agent personas.
        
        Args:
            count: Number of personas to generate
            
        Returns:
            List of persona dictionaries
        """
        personas = []
        
        # Sample names
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", 
                      "Linda", "William", "Elizabeth", "David", "Susan", "Richard", "Jessica", 
                      "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Nancy"]
        
        last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", 
                     "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", 
                     "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"]
        
        # Sample education levels
        education_levels = ["High School", "Bachelor's Degree", "Master's Degree", "PhD", 
                           "No Formal Education", "Associate's Degree", "Technical Training"]
        
        # Sample traits
        trait_pool = ["analytical", "creative", "cautious", "adventurous", "introverted", 
                     "extroverted", "detail-oriented", "big-picture thinker", "emotional", 
                     "logical", "skeptical", "trusting", "conservative", "liberal", 
                     "traditional", "progressive", "practical", "idealistic"]
        
        for i in range(count):
            # Generate random persona
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            age = random.randint(18, 80)
            education = random.choice(education_levels)
            
            # Select 2-4 traits
            num_traits = random.randint(2, 4)
            traits = random.sample(trait_pool, num_traits)
            
            # Cognitive parameters
            openness = random.uniform(0.2, 0.9)
            confirmatory_bias = random.uniform(0.3, 0.8)
            
            # Create persona
            persona = {
                "name": name,
                "age": age,
                "education": education,
                "traits": traits,
                "openness": openness,
                "confirmatory_bias": confirmatory_bias
            }
            
            personas.append(persona)
            
        return personas

    def _initialize_interaction_network(self) -> Dict[int, List[int]]:
        """
        Initialize the interaction network that determines which agents interact.
        Uses a small-world network structure.
        
        Returns:
            Dictionary mapping agent IDs to lists of neighboring agent IDs
        """
        network = {}
        
        # For each agent, connect to 3-5 other agents
        for agent in self.schedule.agents:
            if isinstance(agent, DOAgent):
                # Determine number of connections
                num_connections = random.randint(3, min(5, self.num_agents - 1))
                
                # Create candidate list (all agents except self)
                candidates = [a.unique_id for a in self.schedule.agents 
                             if isinstance(a, DOAgent) and a.unique_id != agent.unique_id]
                
                # Preferential attachment - agents with similar traits more likely to connect
                weights = []
                for candidate_id in candidates:
                    candidate = self.get_agent_by_id(candidate_id)
                    if candidate:
                        similarity = agent._calculate_agent_similarity(candidate)
                        weights.append(similarity)
                    else:
                        weights.append(0.1)  # Default weight
                
                # Normalize weights
                total_weight = sum(weights) or 1  # Avoid division by zero
                weights = [w / total_weight for w in weights]
                
                # Select connections
                connections = []
                for _ in range(num_connections):
                    if candidates:
                        selected_idx = np.random.choice(len(candidates), p=weights)
                        selected = candidates[selected_idx]
                        connections.append(selected)
                        
                        # Remove selected to avoid duplicates
                        weights.pop(selected_idx)
                        candidates.pop(selected_idx)
                        
                        # Renormalize weights
                        total_weight = sum(weights) or 1
                        weights = [w / total_weight for w in weights]
                
                network[agent.unique_id] = connections
        
        return network

    def get_agent_by_id(self, agent_id: int) -> Optional[DOAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The agent ID to look for
            
        Returns:
            The agent object or None if not found
        """
        for agent in self.schedule.agents:
            if agent.unique_id == agent_id:
                return agent
        return None

    def _run_language_game(self) -> None:
        """Run a language game interaction step between connected agents."""
        # For each agent
        for agent in self.schedule.agents:
            if isinstance(agent, DOAgent) and not isinstance(agent, OfficialAgent):
                # Get this agent's connections
                connections = self.interaction_network.get(agent.unique_id, [])
                if not connections:
                    continue
                    
                # Randomly select which connections to interact with (1-3)
                num_interactions = random.randint(1, min(3, len(connections)))
                selected_connections = random.sample(connections, num_interactions)
                
                # Generate utterance if needed
                if not agent.current_utterance:
                    agent.generate_utterance()
                    
                # Share utterance with selected connections
                if agent.current_utterance:
                    for target_id in selected_connections:
                        target_agent = self.get_agent_by_id(target_id)
                        if target_agent and isinstance(target_agent, DOAgent):
                            target_agent.listen(agent.current_utterance)
                
                # Reset current utterance
                agent.current_utterance = None

    def _count_susceptible(self, model) -> int:
        """Count agents in susceptible state."""
        return sum(1 for agent in self.schedule.agents 
                  if isinstance(agent, DOAgent) and not isinstance(agent, OfficialAgent)
                  and agent.belief_state.get_belief_state() == "susceptible")
    
    def _count_infected(self, model) -> int:
        """Count agents in infected state."""
        return sum(1 for agent in self.schedule.agents 
                  if isinstance(agent, DOAgent) and not isinstance(agent, OfficialAgent)
                  and agent.belief_state.get_belief_state() == "infected")
    
    def _count_recovered(self, model) -> int:
        """Count agents in recovered state."""
        return sum(1 for agent in self.schedule.agents 
                  if isinstance(agent, DOAgent) and not isinstance(agent, OfficialAgent)
                  and agent.belief_state.get_belief_state() == "recovered")
    
    def _average_belief_strength(self, model) -> float:
        """Calculate average belief strength across agents."""
        beliefs = [agent.belief_state.belief for agent in self.schedule.agents 
                  if isinstance(agent, DOAgent) and not isinstance(agent, OfficialAgent)]
        return sum(beliefs) / len(beliefs) if beliefs else 0
    
    def _belief_variance(self, model) -> float:
        """Calculate variance in belief strength across agents."""
        beliefs = [agent.belief_state.belief for agent in self.schedule.agents 
                  if isinstance(agent, DOAgent) and not isinstance(agent, OfficialAgent)]
        return np.var(beliefs) if beliefs else 0

    def step(self) -> None:
        """Execute one step of the model."""
        # Increment day counter
        self.day += 1
        
        # Run individual agent steps
        self.schedule.step()
        
        # Run language game interactions
        self._run_language_game()
        
        # Collect data
        self.datacollector.collect(self)
        

# Simplified run function for the model
def run_model(
    num_agents: int = 20,
    fake_news_topic: str = "UFO sighting confirmed by government",
    initial_infected: int = 1,
    intervention_days: List[int] = None,
    steps: int = 30
) -> mesa.model.Model:
    """
    Run the fake news propagation model.
    
    Args:
        num_agents: Number of agents in the simulation
        fake_news_topic: The fake news topic
        initial_infected: Number of initially infected agents
        intervention_days: Days when official interventions occur
        steps: Number of simulation steps to run
        
    Returns:
        The model after running
    """
    if intervention_days is None:
        intervention_days = [5, 15, 25]
        
    # Create model
    model = FPSWorldModel(
        N=num_agents,
        fake_news_topic=fake_news_topic,
        initial_infected_count=initial_infected,
        intervention_days=intervention_days
    )
    
    # Run model
    for _ in range(steps):
        model.step()
        
    return model


# Example usage and analysis function
def analyze_results(model: mesa.model.Model) -> Dict:
    """
    Analyze results from the model run.
    
    Args:
        model: The model after running
        
    Returns:
        Dictionary with analysis results
    """
    # Get agent data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # Get model data
    model_data = model.datacollector.get_model_vars_dataframe()
    
    # Calculate final stats
    final_stats = {
        "final_susceptible": model_data["Susceptible"].iloc[-1],
        "final_infected": model_data["Infected"].iloc[-1],
        "final_recovered": model_data["Recovered"].iloc[-1],
        "max_infected": model_data["Infected"].max(),
        "max_infected_day": model_data["Infected"].argmax(),
        "final_belief_avg": model_data["AverageBeliefStrength"].iloc[-1],
        "final_belief_var": model_data["BeliefVariance"].iloc[-1]
    }
    
    # Analyze impact of interventions
    intervention_days = model.intervention_days
    
    intervention_impact = {}
    for day in intervention_days:
        # Check if we have data for day and day+1
        if day < len(model_data) and day+1 < len(model_data):
            before = model_data["Infected"].iloc[day]
            after = model_data["Infected"].iloc[day+1]
            change = after - before
            pct_change = (change / before) * 100 if before > 0 else 0
            
            intervention_impact[day] = {
                "before": before,
                "after": after,
                "absolute_change": change,
                "percent_change": pct_change
            }
    
    return {
        "final_stats": final_stats,
        "intervention_impact": intervention_impact,
        "model_data": model_data,
        "agent_data": agent_data
    }


if __name__ == "__main__":
    # Example usage
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
        
    print("\nIntervention Impact:")
    for day, impact in results["intervention_impact"].items():
        print(f"- Day {day}: {impact['percent_change']:.2f}% change in infected agents")
        
    # In a Jupyter notebook, you could visualize the results with:
    # import matplotlib.pyplot as plt
    # results["model_data"][["Susceptible", "Infected", "Recovered"]].plot()
    # plt.xlabel("Day")
    # plt.ylabel("Number of Agents")
    # plt.title("Fake News Propagation Dynamics")
    # plt.show()
