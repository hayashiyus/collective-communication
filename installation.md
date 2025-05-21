# Installation Guide

This guide provides instructions for setting up and running the Language Game-Based Fake News Propagation Model.

## Prerequisites

The following software and libraries are required:

- Python 3.7+
- Mesa (agent-based modeling framework)
- NumPy
- Matplotlib (for visualization)
- pandas (for data manipulation)
- (Optional) OpenAI API key for real LLM interactions

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/language-game-fake-news.git
cd language-game-fake-news
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

If the requirements.txt file is not available, install the dependencies manually:

```bash
pip install mesa numpy matplotlib pandas
```

### 4. (Optional) Set Up OpenAI API Access

For production use with real LLM interactions, set up your OpenAI API key:

```bash
# On Windows
set OPENAI_API_KEY=your-api-key

# On macOS/Linux
export OPENAI_API_KEY=your-api-key
```

Alternatively, create a .env file in the project root:

```
OPENAI_API_KEY=your-api-key
```

And then load it in your code with:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Running the Model

### Basic Usage

Run a simple simulation:

```bash
python main.py --mode single --num_agents 20 --topic "Fake news topic" --steps 30
```

### Running Experiments

Run experiments with different intervention schedules:

```bash
python main.py --mode experiment --num_agents 20 --topic "Fake news topic" --steps 30 --repetitions 5 --save_results
```

### Interactive Examples

Explore different examples with visualizations:

```bash
python example.py
```

## Comparing with Original SIS Model (Optional)

If you want to compare with the original SIS model, you'll need to:

1. Clone the original FPS repository:

```bash
git clone https://github.com/LiuYuHan31/FPS.git original_fps
```

2. Make the original model importable (you may need to modify some code)

3. Run comparison experiments:

```bash
python main.py --mode compare --load_file original_results.json --plot_type all
```

## Directory Structure

- `language_game_model.py`: Main implementation of the language game model
- `utils.py`: Utility functions for running experiments and visualizations
- `main.py`: Command-line interface for running simulations
- `example.py`: Example usage scenarios
- `data/`: Directory for storing simulation data
- `results/`: Directory for storing experiment results
- `plots/`: Directory for storing generated plots

## Troubleshooting

- **ImportError**: Make sure all required packages are installed
- **OpenAI API errors**: Check your API key and internet connection
- **Memory errors**: Reduce the number of agents or steps for large simulations
- **Mesa-related errors**: Make sure you're using a compatible version of Mesa

## Next Steps

After installation, we recommend:

1. Start with the example.py file to understand the model capabilities
2. Experiment with different parameters using main.py
3. Explore the code to understand the language game implementation
4. Try creating your own scenarios or extending the model

## License

This project is licensed under the MIT License - see the LICENSE file for details.
