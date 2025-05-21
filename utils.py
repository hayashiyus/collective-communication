"""
Utility functions for running experiments and comparing models.

This module provides helper functions for running simulations, 
visualizing results, and comparing the language game model with the original SIS model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional
import pandas as pd
import json
import os
from language_game_model import run_model as run_language_game_model

# Path configurations
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def run_experiment(
    model_type: str = "language_game",
    num_agents: int = 20,
    fake_news_topic: str = "UFO sighting confirmed by government",
    initial_infected: int = 1,
    intervention_schedules: List[List[int]] = None,
    steps: int = 30,
    repetitions: int = 3,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run an experiment with different intervention schedules and model types.
    
    Args:
        model_type: "language_game" or "sis" (original model)
        num_agents: Number of agents in simulation
        fake_news_topic: The fake news topic
        initial_infected: Number of initially infected agents
        intervention_schedules: List of intervention schedules to test
        steps: Number of simulation steps
        repetitions: Number of repetitions for each configuration
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with experiment results
    """
    # Default intervention schedules if none provided
    if intervention_schedules is None:
        intervention_schedules = [
            [],  # No intervention
            [10],  # Single intervention
            [5, 15],  # Two interventions
            [5, 10, 15]  # Three interventions
        ]
    
    # Initialize results storage
    results = {
        "config": {
            "model_type": model_type,
            "num_agents": num_agents,
            "fake_news_topic": fake_news_topic,
            "initial_infected": initial_infected,
            "steps": steps,
            "repetitions": repetitions
        },
        "schedules": {},
    }
    
    # Try to import original SIS model if needed
    if model_type == "sis":
        try:
            # Assuming the original model is imported as follows
            # Adjust according to actual import requirements
            from fps_original.model import run_model as run_sis_model
            model_runner = run_sis_model
        except ImportError:
            print("WARNING: Original SIS model not found. Defaulting to language game model.")
            model_runner = run_language_game_model
    else:
        model_runner = run_language_game_model
    
    # Run experiments for each intervention schedule
    for schedule in intervention_schedules:
        schedule_key = "_".join(map(str, schedule)) if schedule else "none"
        results["schedules"][schedule_key] = []
        
        print(f"Running with intervention schedule: {schedule}")
        
        # Run multiple repetitions for statistical significance
        for rep in range(repetitions):
            print(f"  Repetition {rep+1}/{repetitions}")
            
            # Run the model
            model = model_runner(
                num_agents=num_agents,
                fake_news_topic=fake_news_topic,
                initial_infected=initial_infected,
                intervention_days=schedule,
                steps=steps
            )
            
            # Extract and store results
            model_data = model.datacollector.get_model_vars_dataframe()
            agent_final = model.datacollector.get_agent_vars_dataframe().xs(steps-1, level="Step")
            
            # Calculate summary statistics
            final_infected = model_data["Infected"].iloc[-1]
            max_infected = model_data["Infected"].max()
            max_infected_day = model_data["Infected"].idxmax()
            final_belief_avg = model_data["AverageBeliefStrength"].iloc[-1] if "AverageBeliefStrength" in model_data else None
            belief_volatility = model_data["BeliefVariance"].mean() if "BeliefVariance" in model_data else None
            
            # Calculate intervention effects
            intervention_effects = []
            for day in schedule:
                if day < len(model_data) and day+1 < len(model_data):
                    before = model_data["Infected"].iloc[day]
                    after = model_data["Infected"].iloc[day+1]
                    effect = {
                        "day": day,
                        "before": float(before),
                        "after": float(after),
                        "change": float(after - before),
                        "percent_change": float(((after - before) / before) * 100) if before > 0 else 0
                    }
                    intervention_effects.append(effect)
            
            # Store repetition results
            rep_result = {
                "final_infected": int(final_infected),
                "max_infected": int(max_infected),
                "max_infected_day": int(max_infected_day),
                "final_susceptible": int(model_data["Susceptible"].iloc[-1]),
                "final_recovered": int(model_data["Recovered"].iloc[-1]),
                "final_belief_avg": float(final_belief_avg) if final_belief_avg is not None else None,
                "belief_volatility": float(belief_volatility) if belief_volatility is not None else None,
                "intervention_effects": intervention_effects,
                "infection_timeline": model_data["Infected"].astype(int).tolist()
            }
            
            results["schedules"][schedule_key].append(rep_result)
    
    # Calculate aggregated statistics
    for schedule_key, reps in results["schedules"].items():
        # Calculate averages across repetitions
        avg_final_infected = sum(r["final_infected"] for r in reps) / len(reps)
        avg_max_infected = sum(r["max_infected"] for r in reps) / len(reps)
        
        # Store aggregated results
        results["schedules"][schedule_key + "_avg"] = {
            "avg_final_infected": avg_final_infected,
            "avg_max_infected": avg_max_infected,
            "std_final_infected": np.std([r["final_infected"] for r in reps]),
            "std_max_infected": np.std([r["max_infected"] for r in reps])
        }
    
    # Save results if requested
    if save_results:
        # Create a meaningful filename
        filename = f"{model_type}_agents{num_agents}_infected{initial_infected}_steps{steps}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    return results

def compare_models(
    results_language_game: Dict[str, Any],
    results_sis: Dict[str, Any],
    plot_type: str = "infected",
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Compare results between language game and SIS models.
    
    Args:
        results_language_game: Results from language game model
        results_sis: Results from SIS model
        plot_type: Type of plot to generate ("infected", "belief", "comparison")
        save_plots: Whether to save plots to disk
        
    Returns:
        Dictionary with comparison metrics
    """
    # Initialize comparison results
    comparison = {
        "metrics": {},
        "plots": {}
    }
    
    # Extract schedule keys present in both results
    common_schedules = set(results_language_game["schedules"].keys()).intersection(
        set(results_sis["schedules"].keys())
    )
    common_schedules = [s for s in common_schedules if not s.endswith("_avg")]
    
    # Compare metrics for each schedule
    for schedule in common_schedules:
        # Get average metrics for language game model
        lg_avg = results_language_game["schedules"].get(schedule + "_avg", {})
        # Get average metrics for SIS model
        sis_avg = results_sis["schedules"].get(schedule + "_avg", {})
        
        # Calculate differences
        if lg_avg and sis_avg:
            diffs = {
                "final_infected_diff": lg_avg["avg_final_infected"] - sis_avg["avg_final_infected"],
                "max_infected_diff": lg_avg["avg_max_infected"] - sis_avg["avg_max_infected"],
                "final_infected_ratio": lg_avg["avg_final_infected"] / max(1, sis_avg["avg_final_infected"]),
                "max_infected_ratio": lg_avg["avg_max_infected"] / max(1, sis_avg["avg_max_infected"])
            }
            
            comparison["metrics"][schedule] = diffs
    
    # Create plots based on plot_type
    if plot_type == "infected" or plot_type == "all":
        fig, axes = plt.subplots(len(common_schedules), 2, figsize=(14, 4 * len(common_schedules)))
        
        for i, schedule in enumerate(common_schedules):
            # Extract infection timelines
            lg_data = results_language_game["schedules"][schedule]
            sis_data = results_sis["schedules"][schedule]
            
            # Calculate average timelines
            lg_timelines = [rep["infection_timeline"] for rep in lg_data]
            sis_timelines = [rep["infection_timeline"] for rep in sis_data]
            
            lg_avg_timeline = np.mean(lg_timelines, axis=0)
            lg_std_timeline = np.std(lg_timelines, axis=0)
            sis_avg_timeline = np.mean(sis_timelines, axis=0)
            sis_std_timeline = np.std(sis_timelines, axis=0)
            
            # Plot language game model
            ax = axes[i, 0] if len(common_schedules) > 1 else axes[0]
            x = range(len(lg_avg_timeline))
            ax.plot(x, lg_avg_timeline, 'b-', label='Average')
            ax.fill_between(x, 
                            lg_avg_timeline - lg_std_timeline, 
                            lg_avg_timeline + lg_std_timeline, 
                            alpha=0.3, color='b')
            ax.set_title(f"Language Game Model - Schedule: {schedule}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Number of Infected Agents")
            ax.legend()
            
            # Plot SIS model
            ax = axes[i, 1] if len(common_schedules) > 1 else axes[1]
            x = range(len(sis_avg_timeline))
            ax.plot(x, sis_avg_timeline, 'r-', label='Average')
            ax.fill_between(x, 
                            sis_avg_timeline - sis_std_timeline, 
                            sis_avg_timeline + sis_std_timeline, 
                            alpha=0.3, color='r')
            ax.set_title(f"SIS Model - Schedule: {schedule}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Number of Infected Agents")
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            filename = f"comparison_infected_timelines.png"
            filepath = os.path.join(PLOTS_DIR, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        comparison["plots"]["infected"] = fig
    
    if plot_type == "comparison" or plot_type == "all":
        # Plot bar chart comparing key metrics
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data for plotting
        schedules = list(common_schedules)
        lg_final = [results_language_game["schedules"][s + "_avg"]["avg_final_infected"] for s in schedules]
        sis_final = [results_sis["schedules"][s + "_avg"]["avg_final_infected"] for s in schedules]
        lg_max = [results_language_game["schedules"][s + "_avg"]["avg_max_infected"] for s in schedules]
        sis_max = [results_sis["schedules"][s + "_avg"]["avg_max_infected"] for s in schedules]
        
        # Plot final infected comparison
        x = np.arange(len(schedules))
        width = 0.35
        axes[0].bar(x - width/2, lg_final, width, label='Language Game')
        axes[0].bar(x + width/2, sis_final, width, label='SIS')
        axes[0].set_xlabel("Intervention Schedule")
        axes[0].set_ylabel("Final Infected Count")
        axes[0].set_title("Final Infected Comparison")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(schedules)
        axes[0].legend()
        
        # Plot max infected comparison
        axes[1].bar(x - width/2, lg_max, width, label='Language Game')
        axes[1].bar(x + width/2, sis_max, width, label='SIS')
        axes[1].set_xlabel("Intervention Schedule")
        axes[1].set_ylabel("Maximum Infected Count")
        axes[1].set_title("Maximum Infected Comparison")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(schedules)
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            filename = f"comparison_metrics.png"
            filepath = os.path.join(PLOTS_DIR, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        comparison["plots"]["comparison"] = fig
    
    if plot_type == "belief" or plot_type == "all":
        # Only applicable to language game model since SIS doesn't track beliefs
        fig, axes = plt.subplots(len(common_schedules), 1, figsize=(10, 4 * len(common_schedules)))
        
        for i, schedule in enumerate(common_schedules):
            # Check if we have agents data with belief values
            if "agent_data" in results_language_game:
                agent_data = results_language_game["agent_data"]
                
                # Plot belief distributions over time
                ax = axes[i] if len(common_schedules) > 1 else axes
                
                # TODO: Implement belief visualization
                # This is a placeholder for belief visualization
                ax.set_title(f"Belief Distribution - Schedule: {schedule}")
                ax.set_xlabel("Step")
                ax.set_ylabel("Average Belief")
                ax.text(0.5, 0.5, "Belief visualization not available", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            filename = f"belief_distributions.png"
            filepath = os.path.join(PLOTS_DIR, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        comparison["plots"]["belief"] = fig
    
    return comparison

def load_results(filename: str) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        filename: Name of results file
        
    Returns:
        Dictionary with experiment results
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "r") as f:
        results = json.load(f)
    return results

def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save experiment results to file.
    
    Args:
        results: Results dictionary
        filename: Name for results file
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Run experiment with language game model
    results_lg = run_experiment(
        model_type="language_game",
        num_agents=20,
        fake_news_topic="A study shows that drinking lemon water cures cancer",
        initial_infected=2,
        intervention_schedules=[[], [10], [5, 15]],
        steps=30,
        repetitions=3
    )
    
    # Note: To run with SIS model, you would need to have the original FPS implementation
    # For demonstration, we'll reuse the language game results
    print("\nLanguage Game Model Results:")
    for schedule, avg_results in [(k, v) for k, v in results_lg["schedules"].items() if k.endswith("_avg")]:
        schedule_name = schedule.replace("_avg", "")
        print(f"  Schedule {schedule_name}:")
        print(f"    Avg Final Infected: {avg_results['avg_final_infected']:.2f}")
        print(f"    Avg Max Infected: {avg_results['avg_max_infected']:.2f}")
    
    print("\nTo compare with SIS model, install the original FPS implementation and run:")
    print("results_sis = run_experiment(model_type='sis', ...)")
    print("compare_models(results_lg, results_sis, plot_type='all')")
