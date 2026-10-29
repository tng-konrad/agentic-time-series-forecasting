# Module 2 promotional copy

## SEO description

Explore routing in agentic forecasting: compare LLM decisions with explicit rules across full, intermittent, short and zero-history sales.

Character count: 138

## LinkedIn promo post

“Forecast the next four weeks.”

For which product? The one with a long sales history? The one that rarely sells? The one with sixty observations? Or the new product with none?

That is the routing problem in Module 2 of Agentic Time Series Forecasting. Before choosing a model, we need a workflow that can use the information actually available.

I built a runnable notebook comparing a simple forecasting baseline, explicit routing rules, and a local LLM router on the same sales evidence. Python handles the fits and scores. The router gets one bounded decision.

The model had a remarkably consistent extra requirement: paperwork.

Across sixteen supported requests, eleven forecasting proposals failed the decision contract and fell back to rules. Five asked unnecessary clarification questions and produced no forecast. Every delivered primary LLM-arm forecast came from the fallback.

There is also a separate successful development call. It demonstrates that an accepted model decision really controls dispatch. It stays outside the benchmark; one working example cannot erase the original run.

The post walks through the four request types, chronological evaluation, coverage, evidence quality, token use and latency. The notebook preserves the responses and recomputes the numerical results offline.

Routing helps make capability boundaries explicit. Whether a language model should make that decision is something we can measure.

Module 2: One forecasting request, several possible problems.

#TimeSeries #Forecasting #AgenticAI #DataScience

## Publication handoff

Use the [article](../a02-post.md) and [executed notebook](../a02-routing.ipynb). Add the public article URL to the LinkedIn draft after publishing; no public URL exists yet. The copy above is a draft and has not been posted.
