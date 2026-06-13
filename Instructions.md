# AI/ML Team Take Home Exercise: Predicting Chess Game Outcomes

## Overview

Using [Chess.com](https://chess.com)'s public API, build a model that predicts the outcome of a chess game (win, loss, or draw for the white player), given the information available before the game starts.

Please don't spend more than 3-4 hours on this exercise. We respect your time and aren't looking for a large-scale model. We want to see how you think about an ML problem: data collection, feature engineering, modeling, and evaluation.

## Data

Your dataset comes from Titled Tuesday events, [Chess.com](https://chess.com)'s weekly blitz tournament for titled players (GMs, IMs, FMs, etc.). Each event is an 11-round Swiss tournament with a 5-minute time control and typically 400+ participants. We suggest using data from the following two recent events:

- February 10, 2026 (455 players, 11 rounds)
  http://api.chess.com/pub/tournament/titled-tuesday-blitz-february-10-2026-6221327

- March 10, 2026 (476 players, 11 rounds)
  http://api.chess.com/pub/tournament/titled-tuesday-blitz-march-10-2026-6277141

Our Published Data API (PubAPI) is public, read-only, and requires no authentication. The tournament endpoint returns metadata about the event, including the list of players and URLs for each round. You can traverse the round and group endpoints to retrieve the individual games, which include player ratings, results, opening codes, PGN, and other fields. Include a User-Agent header with your API requests.

Helpful endpoints:

- **Tournament info:** `/pub/tournament/{id}` — Returns tournament metadata, including players and round URLs
- **Round/group games:** `/pub/tournament/{id}/{round}/{group}` — Returns the games for a specific round/group
- **Player profiles (optional):** `/pub/player/{username}` — Returns player metadata. You may use this to construct pre-game features if you find it useful.
- **Player stats (optional):** `/pub/player/{username}/stats` — Returns rating summaries across game types. You may use this to construct pre-game features if you find it useful.

## What to Build

- **Data collection and feature engineering**
  - Using the tournament data, construct a dataset where each row represents a single game. Your goal is to predict the outcome of the game (win / loss / draw from White's perspective). The API returns several fields per game; not all of them are appropriate as input features.

- **Model**
  - Train a model to predict the game outcome. We're not looking for a leaderboard-topping solution. Instead, we want to see how you frame the problem, what tradeoffs you make, and how you evaluate performance. A simple and well-reasoned model with clear evaluation is better for this exercise than a complex model with unclear metrics.

- **Brief writeup.**
  - In your notebook or README, include:
    - How you split your data and why you chose that approach
    - What you think of the quality of the model (e.g., how does it perform? do you think it is a good model or not?). Again, we're not looking for a leaderboard-topping solution here, but would like an evaluation and opinion.
    - What you would do next with more time

## Deliverables

A GitHub repo (private is fine; just invite us) or zip file containing your code, a notebook, or script that can reproduce your results, and the brief writeup. We should be able to run your code successfully to produce the same results.

## Guidelines

Target time: 3-4 hours. We mean this, don't spend days on this problem. AI coding assistants are welcome, as is anything else that you would use in your day-to-day work. We care that you can explain your choices: the follow-up will be a live walkthrough of your submission.

Use whatever language and libraries you prefer, but note that Python is the default language on the current team.

## Next Steps

After you send your code back to us, we'll review your work and, if selected, we'll invite you to a live follow-up where we'll walk through your code together: your decisions, tradeoffs, and what you'd do differently with more time.

Questions? Reach out to your recruiting contact and we will get back to you promptly.