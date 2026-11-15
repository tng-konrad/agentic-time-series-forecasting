This repo is about agentic workflows for time series problems:
1. A01: From a forecasting notebook to an agentic workflow


   Pattern:  Sequential workflow / prompt chaining.  
   Build a fixed sequence: inspect data → propose candidates → backtest → forecast → explain. Introduce reusable numerical tools and structured outputs between stages. Compare a conventional Python pipeline with an LLM-assisted workflow, measuring forecast consistency, report correctness, and execution overhead.  


2. A02: One forecasting request, several possible problems
   
   
   Pattern: Routing.  
   Introduce established demand, intermittent demand, short histories, and new products. Route each case to an appropriate candidate workflow using measured properties and available metadata. Compare agent routing with explicit rules, including cases where information is missing or ambiguous.  

3. A03: Let the agent choose its next diagnostic

   Pattern: Single-agent tool loop / ReAct-style agent. 
   A local Ollama model chooses diagnostics on three monthly retail series, with six request attempts, five tool attempts and zero retries. Compare every first attempt with a matched fixed checklist; retain unsupported reasons and measured inference overhead.
