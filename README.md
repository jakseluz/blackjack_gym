# Blackjack custom Gymnasium environment game

Project focuses on a Blackjack game implementation using the Python library _Gymnasium_. The game environment can be used for reinforcement learning projects.

Blackjack rules
- [in Polish](https://pl.wikipedia.org/wiki/Blackjack)
- [in English](https://en.wikipedia.org/wiki/Blackjack)

**Note:** The current implementation does not support SPLIT moves. Besides that, most other standard rules apply. There are still areas for improvement - some of them are marked with #TODO comment in the code.

## Requirements

- Python\
  [configuration file](./pyproject.toml) (pyproject.toml)

Install project and its dependencies with:
--
``` bash
pip install -e .
```
- (in the project root directory)

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jakseluz/blackjack_gym.git
   cd blackjack_gym
   ```

2. Check **./notebooks/[main.ipynb](./notebooks/main.ipynb) file** for the tutorial/ information.
3. More detailed analysis available (in Polish) in **./notebooks/[report.ipynb](./notebooks/report.ipynb)** 

## Authors
- **Jakub Łabuz** ([jakseluz](https://github.com/jakseluz)) - Blackjack
- **Maciej Maj** ([mmaj1](https://github.com/mmaj1)) - tests and report

## Used resources:
- [Playing Card Deck](https://opengameart.org/content/bridge-sized-playing-card-deck-png-cc0)
- [Gymnasium Environment Template](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

---

_For questions or contributions, please open an issue or submit a pull request!_  

4. Project report (in Polish by Maciej Maj) (https://www.overleaf.com/project/6a01078b024e8d1800c6078a)