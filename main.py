"""
Main script for running the fake news propagation simulation.

This script provides a command-line interface for running the fake news
propagation simulation with different parameters and configurations.
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
from language_game_model import run_model, analyze_results
from utils import run_experiment, compare_models, load_results, save_results

def main():
    """Main function to run the simulation from command line."""
    parser = argparse.ArgumentParser(description='Run fake news propagation simulation')
    
    # Define command-line arguments
    parser.add_argument('--mode', choices=['single', 'experiment', 'compare'], default='single',
                        help='Mode to run: single simulation, experiment with multiple configs, or compare models')
    
    parser.add_argument('--model_type', choices=['language_game', 'sis'], default='language_game',
                        help='Model type: language_game (new) or sis (original)')
    
    parser.add_argument('--num_agents', type=int, default=20,
                        help='Number of agents in the simulation')
    
    parser.add_argument('--topic', type=str, default="UFO sighting confirmed by government",
                        help='Fake news topic for the simulation')
    
    parser.add_argument('--initial_infected', type=int, default=1,
                        help='Number of initially infected agents')
    
    parser.add_argument('--intervention_days', type=str, default="5,15,25",
                        help='Comma-separated list of intervention days')
    
    parser.add_argument('--steps', type=int, default=30,
                        help='Number of simulation steps')
    
    parser.add_argument('--repetitions', type=int, default=3,
                        help='Number of repetitions for experiments')
    
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to file')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--load_file', type=str,
                        help='Load results from file for comparison')
    
    parser.add_argument('--plot_type', choices=['infected', 'belief', 'comparison', 'all'], 
                        default='all', help='Type of plots to generate')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse intervention days
    if args.intervention_days:
        intervention_days = [int(day) for day in args.intervention_days.split(',')]
    else:
        intervention_days = []
    
    # Run based on selected mode
    if args.mode == 'single':
        # Run a single simulation
        print(f"Running single simulation with {args.model_type} model")
        print(f"Parameters: {args.num_agents} agents, {args.initial_infected} initially infected")
        print(f"Fake news topic: '{args.topic}'")
        print(f"Intervention days: {intervention_days}")
        print(f"Running for {args.steps} steps")
        
        # Run model
        model = run_model(
            num_agents=args.num_agents,
            fake_news_topic=args.topic,
            initial_infected=args.initial_infected,
            intervention_days=intervention_days,
            steps=args.steps
        )
        
        # Analyze and print results
        results = analyze_results(model)
        
        print("\nResults Summary:")
        print("---------------")
        for key, value in results["final_stats"].items():
            print(f"- {key}: {value}")
            
        # Print intervention impact
        print("\nIntervention Impact:")
        for day, impact in results["intervention_impact"].items():
            print(f"- Day {day}: {impact['percent_change']:.2f}% change in infected agents")
        
        # Save results if requested
        if args.save_results:
            filename = f"single_{args.model_type}_{args.num_agents}agents_{args.steps}steps.json"
            filepath = os.path.join(args.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {filepath}")
            
        # Generate plots
        model_data = results["model_data"]
        
        # Plot state counts over time
        plt.figure(figsize=(10, 6))
        model_data[["Susceptible", "Infected", "Recovered"]].plot()
        plt.xlabel("Day")
        plt.ylabel("Number of Agents")
        plt.title("Fake News Propagation Dynamics")
        plt.grid(True)
        
        # Save plot if requested
        if args.save_results:
            plot_filename = f"single_{args.model_type}_{args.num_agents}agents_{args.steps}steps_dynamics.png"
            plot_filepath = os.path.join(args.output_dir, plot_filename)
            plt.savefig(plot_filepath)
            print(f"Plot saved to {plot_filepath}")
        
        plt.show()
        
    elif args.mode == 'experiment':
        # Run experiment with multiple configurations
        print(f"Running experiment with {args.model_type} model")
        print(f"Parameters: {args.num_agents} agents, {args.initial_infected} initially infected")
        print(f"Fake news topic: '{args.topic}'")
        print(f"Running for {args.steps} steps with {args.repetitions} repetitions")
        
        # Create different intervention schedules
        intervention_schedules = [
            [],  # No intervention
            [args.steps // 2],  # Single intervention in the middle
            [args.steps // 3, 2 * args.steps // 3],  # Two interventions
            intervention_days  # User-specified schedule
        ]
        
        # Run experiment
        experiment_results = run_experiment(
            model_type=args.model_type,
            num_agents=args.num_agents,
            fake_news_topic=args.topic,
            initial_infected=args.initial_infected,
            intervention_schedules=intervention_schedules,
            steps=args.steps,
            repetitions=args.repetitions,
            save_results=args.save_results
        )
        
        # Print summary of results
        print("\nExperiment Results Summary:")
        print("---------------------------")
        
        for schedule, avg_results in [(k, v) for k, v in experiment_results["schedules"].items() if k.endswith("_avg")]:
            schedule_name = schedule.replace("_avg", "")
            print(f"Schedule {schedule_name}:")
            print(f"  Avg Final Infected: {avg_results['avg_final_infected']:.2f}")
            print(f"  Avg Max Infected: {avg_results['avg_max_infected']:.2f}")
        
    elif args.mode == 'compare':
        # Compare models
        if not args.load_file:
            print("Error: --load_file required for compare mode")
            return
        
        print(f"Comparing models: current {args.model_type} vs loaded model")
        
        # First run the current model
        print(f"Running {args.model_type} model...")
        current_results = run_experiment(
            model_type=args.model_type,
            num_agents=args.num_agents,
            fake_news_topic=args.topic,
            initial_infected=args.initial_infected,
            steps=args.steps,
            repetitions=args.repetitions,
            save_results=False
        )
        
        # Load the comparison model
        print(f"Loading results from {args.load_file}...")
        loaded_results = load_results(args.load_file)
        
        # Compare results
        print("Comparing results...")
        comparison = compare_models(
            results_language_game=current_results if args.model_type == "language_game" else loaded_results,
            results_sis=loaded_results if args.model_type == "language_game" else current_results,
            plot_type=args.plot_type,
            save_plots=args.save_results
        )
        
        # Print comparison summary
        print("\nComparison Summary:")
        print("------------------")
        
        for schedule, metrics in comparison["metrics"].items():
            print(f"Schedule {schedule}:")
            print(f"  Final Infected Diff: {metrics['final_infected_diff']:.2f}")
            print(f"  Final Infected Ratio: {metrics['final_infected_ratio']:.2f}")
            print(f"  Max Infected Diff: {metrics['max_infected_diff']:.2f}")
            print(f"  Max Infected Ratio: {metrics['max_infected_ratio']:.2f}")
        
        # Show plots
        for plot_name, fig in comparison["plots"].items():
            plt.figure(fig.number)
            plt.show()
    
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()
