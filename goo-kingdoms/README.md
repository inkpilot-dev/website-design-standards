# Goo Kingdoms

A 4X strategy game you watch. Four factions of blobs fight over a 20 by 20 island in full 3D. Every faction is played by Claude, and every blob is its own Claude agent. Each day, each blob reads only what it can see, what its team remembers and what it heard nearby, then picks one action.

| Faction | Doctrine |
| --- | --- |
| Crimson Claude, The Tacticians | Balanced expansion, disciplined diplomacy, coordinated raids |
| Cobalt Claude, The Architects | Economy, development, fast research, towers and walls |
| Amber Claude, The Merchants | Coins, loot, harbors and gunboats, paid alliances |
| Emerald Claude, The Vanguard | Scouting, traps, boss hunting, veteran and champion blobs |

## Run it

You need Node.js 22.18 or newer.

```bash
npm install
```

```bash
npm start
```

Open http://localhost:3000.

With no API key, the built-in local bot plays, so the game always runs. The local bot is a rule-based imitation of each doctrine. It is not Claude, and the stats page keeps its matches separate.

### Let Claude play

1. Copy `.env.example` to `.env`.
2. Put your Anthropic API key in `ANTHROPIC_API_KEY`.
3. Restart with `npm start`.

You can also paste a key in **Settings** in the app. The key stays in the server's memory on your machine; the app never writes it to disk or sends it to the browser.

Claude mode is the only thing that talks to the network: one request to the Anthropic API per living blob per day. Defaults:

- **Model:** Claude Haiku 5.5. You can switch to Sonnet 5.5 or Opus 5.5 in Settings.
- **Effort:** low.
- **Concurrency:** 8 requests in parallel.
- **Budget:** $5 per match. The match pauses when its estimated spend reaches the budget.

Spend estimate at about 3,500 input and 400 output tokens per blob per day:

| Model | 12 blobs, one day | 100 days with about 30 blobs |
| --- | --- | --- |
| Haiku 5.5 | about $0.007 | about $1.6 |
| Sonnet 5.5 | about $0.13 | about $33 |
| Opus 5.5 | about $0.26 | about $66 |

Prompt caching of each faction's system prompt brings these down. The status chip in the top bar shows the running estimate.

## Watching

- **Camera:**
  - Drag to orbit, right-drag to pan, scroll to zoom.
  - W A S D fly over the island, and Q and E move down and up.
  - Double-click or press F to follow the selected unit. Home resets the view.
- **Play controls:** play, pause, step one day, and speeds of 1x, 2x, 5x and Max. Space plays and pauses; the full stop key steps one day.
- **Seed:** enter a seed and press New match. The same seed always gives the same island.
- **View:**
  - Spectator sees everything, including armed traps.
  - Each faction's fog of war.
  - A single blob's sight, with its hearing range marked.
- **Inspector:** click any blob, city, building, loot pile or the boss. For a blob you see its stats, gear, rank, the day's action and whether it worked, Claude's reasoning, what it said, tokens used and its recent days.
- **Replay:** drag the replay slider to any earlier day, or download the whole match as JSON.
- **Rulebook:** every rule and number, generated from the engine's constants.
- **Faction stats:** per-faction wins, territory, kills and deaths, economy, payments, speech, alliances, valid-action rate, API errors and tokens per decision, across every finished match on this machine.

## Rules in brief

- **Start:** each team has a capital, 3 blobs and 15 coins.
- **A day:** blobs act in a freshly shuffled order through five phases. Combat damage lands all at once, so mutual kills happen, and captures resolve last.
- **Economy:**
  - Cities grow levels from farms, lumber camps, mines and harbors, and pay coins every day.
  - Seven techs unlock gear, buildings and boats.
  - A hidden trap hurts an enemy that steps on it, and a tower shoots enemies in range.
- **Diplomacy:** alliances and payments between teams. Blobs can speak, and every blob within 3 tiles hears it, friend or foe.
- **World:** loot washes up every 12 days, and a marsh boss rises mid-game.
- **Losing cities:** a team that loses every city fights on as wanderers with no income.
- **Victory:** last team standing, or holding two thirds of all cities for 20 days in a row, or the most cities at the day limit (300 by default, up to 1000).

The **Rulebook** tab in the app has every detail.

## Project layout

```
src/shared/     types, rules (every number), seeded RNG, grid helpers
src/sim/        map generation, the day-resolution engine, per-blob observations
src/agents/     faction personas, the JSON action schema, the Claude runner, the local bot
src/server/     HTTP server, live event stream, match controller, results store
src/client/     3D renderer (Three.js), UI, rulebook, stats
public/         page shell and styles (the client bundle is built into public/app.js)
```

Other commands:

```bash
npm run typecheck
```

```bash
npm run headless -- 10 300 1
```

The second command plays 10 local-bot matches of up to 300 days from seed 1 and prints a summary.
