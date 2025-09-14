# Travel Planning AI Agent

This is a Python-based AI agent built with LangGraph that provides intelligent travel planning assistance with a focus on eco-friendly options and CO2 emission calculations.

## Features

- **Intelligent Travel Planning**: Uses LangGraph workflow to process natural language queries
- **CO2 Emission Calculations**: Compares different travel modes and their environmental impact
- **Eco-Friendly Recommendations**: Prioritizes sustainable travel options
- **City Information**: Provides hotel and attraction recommendations
- **Real-time Distance Calculation**: Integrates with Google Maps API
- **Database Integration**: Uses SQLite database for city and travel data

## Architecture

The agent follows a LangGraph workflow with the following components:

1. **Query Parser**: Extracts travel information from natural language
2. **Data Retrieval**: Gets travel data, distances, and CO2 calculations
3. **Response Generation**: Uses OpenAI GPT to generate helpful responses

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Google Maps API key (optional)

### Installation

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

4. **Start the agent**:
   ```bash
   python start_agent.py
   ```

## Configuration

Edit the `.env` file with your API keys:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
DATABASE_PATH=../travel_data.db
AGENT_MODEL=gpt-3.5-turbo
AGENT_TEMPERATURE=0.7
```

## API Endpoints

- `POST /chat` - Process chat messages
- `GET /cities` - Get available cities
- `GET /travel-modes` - Get travel modes and emission factors
- `POST /compare-route` - Compare travel options for a route
- `GET /health` - Health check

## Usage Examples

### Chat Interface
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the best way to travel from Delhi to Mumbai?"}'
```

### Route Comparison
```bash
curl -X POST "http://localhost:8000/compare-route" \
  -H "Content-Type: application/json" \
  -d '{"source": "Delhi", "destination": "Mumbai"}'
```

## Travel Modes Supported

- **Air Travel**: Flight (average)
- **Road Transport**: Diesel Car, Petrol Car, Electric Car
- **Public Transport**: Train (diesel/electric), Bus (shared)
- **Active Transport**: Bicycle, Walking

## CO2 Emission Factors

The agent uses standardized emission factors (kg CO2 per km):

- Flight: 0.255 kg/km
- Petrol Car: 0.192 kg/km
- Diesel Car: 0.171 kg/km
- Electric Car: 0.053 kg/km
- Train: 0.041 kg/km
- Bus: 0.089 kg/km
- Bicycle/Walking: 0.0 kg/km

## Integration

The agent integrates with:

- **Frontend**: React chat interface
- **Backend**: Node.js API server
- **Database**: SQLite with travel data
- **External APIs**: Google Maps, OpenAI

## Development

### Project Structure

```
agent/
├── config/          # Configuration settings
├── tools/           # Travel calculation tools
├── workflows/       # LangGraph workflows
├── utils/           # Database utilities
├── main.py          # FastAPI server
├── start_agent.py   # Startup script
└── requirements.txt # Dependencies
```

### Adding New Features

1. **New Travel Modes**: Add to `config/settings.py`
2. **New Tools**: Create in `tools/` directory
3. **Workflow Changes**: Modify `workflows/travel_agent.py`

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Ensure it's set in `.env`
2. **Database Path**: Check if `travel_data.db` exists
3. **Port Conflicts**: Ensure port 8000 is available
4. **Dependencies**: Run `pip install -r requirements.txt`

### Logs

Check the console output for detailed error messages and debugging information.

## License

MIT License - see LICENSE file for details.
