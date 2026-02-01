# Stock Analysis Multi-Agent Platform

A comprehensive multi-agent system for analyzing key factors that impact stock performance. The platform uses a Master-Servant architecture where a Master Agent orchestrates multiple specialized Servant Agents to provide holistic stock analysis.

## Architecture

```
                    ┌─────────────────────┐
                    │    Master Agent     │
                    │   (Orchestrator)    │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Fundamental    │ │    Technical     │ │    Sentiment     │
│   Analysis       │ │    Analysis      │ │    Analysis      │
│   Agent          │ │    Agent         │ │    Agent         │
└──────────────────┘ └──────────────────┘ └──────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Risk Assessment │
                    │      Agent       │
                    └──────────────────┘
```

## Features

### Master Agent
- **Orchestrates** all servant agents (parallel or sequential execution)
- **Aggregates** results from all agents
- **Generates** overall score and recommendation
- **Produces** executive summary with key insights

### Servant Agents

#### 1. Fundamental Analysis Agent
Analyzes company financials and valuation metrics:
- Valuation ratios (P/E, P/B, P/S, PEG)
- Profitability metrics (margins, ROE, ROA)
- Growth metrics (revenue, earnings growth)
- Financial health (debt ratios, liquidity)
- Dividend analysis

#### 2. Technical Analysis Agent
Analyzes price patterns and technical indicators:
- Moving Averages (SMA, EMA)
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Analysis (OBV)
- Trend Detection (ADX)
- Buy/Sell Signal Generation

#### 3. Sentiment Analysis Agent
Analyzes market sentiment and news:
- News article sentiment analysis
- Analyst ratings and recommendations
- Price target analysis
- Overall sentiment scoring

#### 4. Risk Assessment Agent
Evaluates risk factors and volatility:
- Volatility metrics (daily, annualized)
- Beta coefficient
- Risk ratios (Sharpe, Sortino)
- Value at Risk (VaR)
- Maximum drawdown analysis
- Risk factor identification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/agentic-fii.git
cd agentic-fii
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

Analyze a single stock:
```bash
python main.py AAPL
```

Analyze multiple stocks:
```bash
python main.py AAPL GOOGL MSFT AMZN
```

Output as JSON:
```bash
python main.py AAPL --json
```

Run agents sequentially:
```bash
python main.py AAPL --sequential
```

### Python API Usage

```python
import asyncio
from src.agents.master_agent import MasterAgent

async def analyze():
    # Create the master agent
    master = MasterAgent(execution_mode="parallel")

    # Analyze a stock
    report = await master.analyze("AAPL")

    # Access results
    print(f"Overall Score: {report.overall_score}")
    print(f"Recommendation: {report.recommendation}")
    print(f"Key Strengths: {report.key_strengths}")
    print(f"Key Risks: {report.key_risks}")

asyncio.run(analyze())
```

### Selective Agent Execution

Run only specific agents:
```python
# Only technical analysis
report = await master.analyze("AAPL", agents_to_run=["technical"])

# Fundamental and risk analysis
report = await master.analyze("AAPL", agents_to_run=["fundamental", "risk"])
```

### Custom Weights

Configure custom weights for the overall score:
```python
master = MasterAgent(
    execution_mode="parallel",
    weights={
        "fundamental": 0.35,  # 35% weight
        "technical": 0.25,    # 25% weight
        "sentiment": 0.15,    # 15% weight
        "risk": 0.25,         # 25% weight
    }
)
```

## Project Structure

```
agentic-fii/
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration settings
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Abstract base agent
│   │   ├── master_agent.py     # Orchestrator agent
│   │   ├── fundamental_agent.py
│   │   ├── technical_agent.py
│   │   ├── sentiment_agent.py
│   │   └── risk_agent.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
└── examples/
    ├── __init__.py
    ├── basic_usage.py
    ├── custom_agents.py
    ├── selective_analysis.py
    └── batch_analysis.py
```

## Creating Custom Agents

You can create custom agents by extending the `BaseAgent` class:

```python
from src.agents.base_agent import BaseAgent
from typing import Any

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyAgent", agent_type="custom")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        # Implement your analysis logic here
        return {
            "data": {"key": "value"},
            "score": 75.0,
            "summary": "Analysis complete"
        }

# Add to master agent
master = MasterAgent()
master.add_agent("custom", MyCustomAgent(), weight=0.10)
```

## Configuration

Configuration can be done via environment variables or a `.env` file:

```env
# Execution mode
EXECUTION_MODE=parallel

# Agent weights
WEIGHT_FUNDAMENTAL=0.30
WEIGHT_TECHNICAL=0.25
WEIGHT_SENTIMENT=0.20
WEIGHT_RISK=0.25

# Logging
LOG_LEVEL=INFO

# Optional API keys
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

## Output Example

```
======================================================================
  STOCK ANALYSIS REPORT: AAPL
  Apple Inc.
======================================================================

📊 CURRENT MARKET DATA
----------------------------------------
  Current Price:    $178.50
  Previous Close:   $177.25
  Market Cap:       $2.78T
  Volume:           52,345,678

🎯 OVERALL ANALYSIS
----------------------------------------
  Overall Score:    [████████████████░░░░] 72.5/100
  Recommendation:   BUY
  Confidence:       85%

📈 ANALYSIS BREAKDOWN
----------------------------------------
  ✅ Fundamental     Score:   75.0
  ✅ Technical       Score:   68.5
  ✅ Sentiment       Score:   71.0
  ✅ Risk            Score:   74.0

🔍 KEY INSIGHTS
----------------------------------------
  Strengths:
    ✓ Strong profit margins
    ✓ Bullish technical trend
    ✓ Positive market sentiment
  Risks:
    ⚠ Premium valuation
    ⚠ High market sensitivity

======================================================================
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
