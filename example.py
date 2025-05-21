"""
Example usage of the language game-based fake news propagation model.

This script demonstrates how to use the language game model and compare it
with the original SIS model (if available).
"""

import matplotlib.pyplot as plt
import numpy as np
from language_game_model import run_model, analyze_results
from utils import run_experiment, compare_models

# Example 1: Basic usage of language game model
def example_basic_usage():
    """Demonstrate basic usage of the language game model."""
    print("\n=== Example 1: Basic Usage ===")
    
    # Run a simple simulation
    model = run_model(
        num_agents=20,
        fake_news_topic="A study shows that drinking lemon water cures cancer",
        initial_infected=2,
        intervention_days=[5, 15, 25],
        steps=30
    )
    
    # Analyze results
    results = analyze_results(model)
    
    # Print summary
    print("\nResults Summary:")
    for key, value in results["final_stats"].items():
        print(f"- {key}: {value}")
        
    # Print intervention impact
    print("\nIntervention Impact:")
    for day, impact in results["intervention_impact"].items():
        print(f"- Day {day}: {impact['percent_change']:.2f}% change in infected agents")
    
    # Plot state counts over time
    model_data = results["model_data"]
    plt.figure(figsize=(10, 6))
    model_data[["Susceptible", "Infected", "Recovered"]].plot()
    plt.xlabel("Day")
    plt.ylabel("Number of Agents")
    plt.title("Fake News Propagation Dynamics")
    plt.grid(True)
    plt.show()
    
    # Plot belief strength over time
    plt.figure(figsize=(10, 6))
    model_data["AverageBeliefStrength"].plot(label="Average Belief")
    plt.plot(model_data.index, np.ones(len(model_data)) * 0.5, 'k--', label="Neutral (0.5)")
    plt.xlabel("Day")
    plt.ylabel("Belief Strength")
    plt.title("Evolution of Average Belief Strength")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example 2: Experiment with different intervention schedules
def example_intervention_experiment():
    """Demonstrate experimenting with different intervention schedules."""
    print("\n=== Example 2: Intervention Experiment ===")
    
    # Define intervention schedules to test
    intervention_schedules = [
        [],  # No intervention
        [15],  # Single intervention mid-way
        [10, 20],  # Two interventions
        [5, 15, 25]  # Three interventions
    ]
    
    # Run experiment
    results = run_experiment(
        model_type="language_game",
        num_agents=20,
        fake_news_topic="A study shows that 5G towers cause health problems",
        initial_infected=2,
        intervention_schedules=intervention_schedules,
        steps=30,
        repetitions=2,  # Use low value for demonstration, increase for real experiments
        save_results=False
    )
    
    # Print summary
    print("\nExperiment Results Summary:")
    for schedule, avg_results in [(k, v) for k, v in results["schedules"].items() if k.endswith("_avg")]:
        schedule_name = schedule.replace("_avg", "")
        print(f"\nSchedule {schedule_name}:")
        print(f"  Avg Final Infected: {avg_results['avg_final_infected']:.2f}")
        print(f"  Avg Max Infected: {avg_results['avg_max_infected']:.2f}")
    
    # Plot comparison of final infected for different schedules
    schedules = [s for s in results["schedules"].keys() if not s.endswith("_avg")]
    final_infected = [results["schedules"][s + "_avg"]["avg_final_infected"] for s in schedules]
    max_infected = [results["schedules"][s + "_avg"]["avg_max_infected"] for s in schedules]
    
    plt.figure(figsize=(12, 6))
    
    # Plot final infected
    plt.subplot(1, 2, 1)
    plt.bar(range(len(schedules)), final_infected)
    plt.xlabel("Intervention Schedule")
    plt.ylabel("Final Infected Count")
    plt.title("Effect of Intervention Schedule on Final Infected")
    plt.xticks(range(len(schedules)), schedules)
    
    # Plot max infected
    plt.subplot(1, 2, 2)
    plt.bar(range(len(schedules)), max_infected)
    plt.xlabel("Intervention Schedule")
    plt.ylabel("Maximum Infected Count")
    plt.title("Effect of Intervention Schedule on Max Infected")
    plt.xticks(range(len(schedules)), schedules)
    
    plt.tight_layout()
    plt.show()

# Example 3: Examining agent belief dynamics
def example_belief_dynamics():
    """Demonstrate examining individual agent belief dynamics."""
    print("\n=== Example 3: Agent Belief Dynamics ===")
    
    # Run model with fewer agents for clarity
    model = run_model(
        num_agents=10,
        fake_news_topic="A private company has built a working fusion reactor",
        initial_infected=2,
        intervention_days=[10, 20],
        steps=30
    )
    
    # Get agent data from datacollector
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # Plot belief evolution for each agent
    plt.figure(figsize=(12, 8))
    
    # Extract unique agent IDs
    agent_ids = sorted(set(agent_data.index.get_level_values('AgentID')))
    
    # Filter out official agent and other non-DOA agents
    agent_ids = [aid for aid in agent_ids if aid < model.num_agents]
    
    # Create a colormap for agents
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_ids)))
    
    # Plot belief evolution for each agent
    for i, agent_id in enumerate(agent_ids):
        agent_beliefs = agent_data.xs(agent_id, level="AgentID")["Belief"]
        plt.plot(agent_beliefs, color=colors[i], label=f"Agent {agent_id}")
    
    # Plot intervention days
    for day in model.intervention_days:
        plt.axvline(x=day, color='r', linestyle='--', alpha=0.5)
        plt.text(day, 0.95, f"Intervention", rotation=90, verticalalignment='top')
    
    # Add neutral line
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label="Neutral")
    
    plt.xlabel("Step")
    plt.ylabel("Belief Strength")
    plt.title("Evolution of Individual Agent Beliefs")
    plt.grid(True, alpha=0.3)
    
    # Add a legend or annotations
    plt.text(0, 1.05, "Belief = 1.0: Fully believes fake news", 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0, 1.02, "Belief = 0.0: Completely rejects fake news", 
             transform=plt.gca().transAxes, fontsize=10)
    
    # Add compact legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Plot belief distribution at different time points
    time_points = [0, 10, 20, 29]  # Start, before/after interventions, end
    
    plt.figure(figsize=(14, 10))
    for i, time in enumerate(time_points):
        plt.subplot(2, 2, i+1)
        
        # Get beliefs at this time point
        beliefs = [agent_data.xs((aid, time), level=("AgentID", "Step"))["Belief"] 
                  for aid in agent_ids]
        
        # Plot histogram
        plt.hist(beliefs, bins=10, range=(0, 1))
        plt.xlabel("Belief Strength")
        plt.ylabel("Number of Agents")
        plt.title(f"Belief Distribution at Step {time}")
    
    plt.tight_layout()
    plt.show()

# Example 4: Understanding active inference dynamics
def example_active_inference():
    """Demonstrate the active inference dynamics in agent decision-making."""
    print("\n=== Example 4: Active Inference Dynamics ===")
    print("This example would require modifications to the model code to track and visualize")
    print("the free energy dynamics and action selection probabilities.")
    print("\nFor a real implementation, you would need to:")
    print("1. Add tracking of expected free energy for each action")
    print("2. Record action selection frequencies")
    print("3. Monitor belief updates and free energy minimization")
    
    # Note: This is a placeholder. To fully implement this example, 
    # you would need to extend the model code to track free energy metrics.
    
    print("\nAs a simplified demonstration, we'll just run a model and")
    print("provide conceptual explanation of active inference principles:")
    
    # Run a simple model
    model = run_model(
        num_agents=5,  # Small number for clarity
        fake_news_topic="New AI algorithm can predict stock market with 100% accuracy",
        initial_infected=1,
        intervention_days=[10],
        steps=20
    )
    
    # Conceptual explanation
    print("\nActive Inference Principles in the Model:")
    print("---------------------------------------")
    print("1. Agents select actions to minimize expected free energy")
    print("2. Free energy has two components:")
    print("   - Prediction error (difference between beliefs and evidence)")
    print("   - Complexity cost (difference from prior beliefs)")
    print("3. Action selection balances:")
    print("   - Epistemic value (information gain)")
    print("   - Pragmatic value (goal-directed behavior)")
    print("4. This leads to adaptive behavior like:")
    print("   - More listening when uncertainty is high")
    print("   - More sharing when certainty is high")
    print("   - More questioning for beliefs in the middle range")
    print("   - More reflection when facing contradictory evidence")

# Example 5: Comparing language game model with SIS model
def example_model_comparison():
    """Demonstrate comparing language game model with SIS model."""
    print("\n=== Example 5: Model Comparison ===")
    
    # Check if original SIS model is available
    try:
        # Attempt to import the original model
        # This is just a placeholder - would need the actual import
        from fps_original.model import run_model as run_sis_model
        sis_available = True
    except ImportError:
        sis_available = False
    
    if not sis_available:
        print("Original SIS model not available. This example requires the original")
        print("FPS implementation. For demonstration, we'll simulate the comparison")
        print("by using the language game model with different parameters.")
        
        # For demonstration, we'll run language game model twice with different parameters
        # This is NOT a real comparison, just for demonstration
        
        print("\nRunning language game model with standard parameters...")
        results_lg = run_experiment(
            model_type="language_game",
            num_agents=20,
            fake_news_topic="New evidence suggests ancient aliens built the pyramids",
            initial_infected=2,
            intervention_schedules=[[], [10], [5, 15]],
            steps=30,
            repetitions=2,
            save_results=False
        )
        
        print("\nSimulating SIS model using modified parameters...")
        # This is just a simulation! Not a real SIS model
        results_sis_sim = run_experiment(
            model_type="language_game",  # Still language game but with different params
            num_agents=20,
            fake_news_topic="New evidence suggests ancient aliens built the pyramids",
            initial_infected=3,  # More initial infected to simulate different dynamics
            intervention_schedules=[[], [10], [5, 15]],
            steps=30,
            repetitions=2,
            save_results=False
        )
        
        print("\nComparing models (simulated comparison)...")
        comparison = compare_models(
            results_language_game=results_lg,
            results_sis=results_sis_sim,
            plot_type="infected",
            save_plots=False
        )
        
        print("\nNote: This is a simulated comparison for demonstration only!")
        print("To perform a real comparison, you need the original SIS model implementation.")
        
    else:
        print("\nRunning language game model...")
        results_lg = run_experiment(
            model_type="language_game",
            num_agents=20,
            fake_news_topic="New evidence suggests ancient aliens built the pyramids",
            initial_infected=2,
            intervention_schedules=[[], [10], [5, 15]],
            steps=30,
            repetitions=3,
            save_results=False
        )
        
        print("\nRunning original SIS model...")
        results_sis = run_experiment(
            model_type="sis",
            num_agents=20,
            fake_news_topic="New evidence suggests ancient aliens built the pyramids",
            initial_infected=2,
            intervention_schedules=[[], [10], [5, 15]],
            steps=30,
            repetitions=3,
            save_results=False
        )
        
        print("\nComparing models...")
        comparison = compare_models(
            results_language_game=results_lg,
            results_sis=results_sis,
            plot_type="all",
            save_plots=False
        )
        
        # Print comparison summary
        print("\nComparison Summary:")
        for schedule, metrics in comparison["metrics"].items():
            print(f"\nSchedule {schedule}:")
            print(f"  Final Infected Diff: {metrics['final_infected_diff']:.2f}")
            print(f"  Final Infected Ratio: {metrics['final_infected_ratio']:.2f}")
            print(f"  Max Infected Diff: {metrics['max_infected_diff']:.2f}")
            print(f"  Max Infected Ratio: {metrics['max_infected_ratio']:.2f}")

if __name__ == "__main__":
    print("Running examples of language game-based fake news propagation model")
    print("------------------------------------------------------------------")
    
    # Run examples
    example_basic_usage()
    example_intervention_experiment()
    example_belief_dynamics()
    example_active_inference()
    example_model_comparison()
    
    print("\nAll examples completed!")
