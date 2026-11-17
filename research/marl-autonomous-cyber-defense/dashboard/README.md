# Sentinel MARL Research Console

A responsive, dependency-free research dashboard preview for the autonomous cyber-defense project.

## Run locally

From this directory, start any static web server, for example:

```bash
python -m http.server 8080
```

Open http://localhost:8080.

## Current scope

- Responsive dark research-console layout.
- Overview cards, illustrative reward chart, five-agent fleet, research readiness gates and activity feed.
- Navigation state, mock episode preview interaction, experiment configuration preview and JSON snapshot export.
- No frontend build step or third-party JavaScript dependency.

## Important

This is a UI prototype. All dashboard metrics are illustrative demo data. The Run button does not train a model, and no CybORG environment, model checkpoint, API, or experiment database is connected. Live integration requires backend endpoints, authentication, experiment lifecycle APIs, and validated telemetry schemas.
