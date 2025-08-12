# β-Lactam Adjuvant Discovery Dashboard

A modern, interactive dashboard for exploring β-lactam adjuvant predictions and knowledge graph results.

## Features

- **Overview Cards**: Key metrics and statistics at a glance
- **Model Performance**: Visual charts showing MRR, Hits@K, and AUROC
- **Predictions Table**: Interactive table of top adjuvant candidates with filtering
- **Knowledge Graph Visualization**: Interactive Cytoscape.js graph of the knowledge graph
- **Data Source Distribution**: Charts showing data contribution from different sources
- **Real-time API**: FastAPI backend serving pipeline results

## Tech Stack

- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **UI Components**: shadcn/ui components
- **Charts**: Recharts for data visualization
- **Graph**: Cytoscape.js for knowledge graph visualization
- **Backend**: FastAPI with CORS support
- **Styling**: Tailwind CSS with custom design system

## Quick Start

### Prerequisites

- Node.js 18+ 
- Python 3.8+ (for FastAPI backend)
- Pipeline results in `../results/` directory

### Installation

1. Install dependencies:
```bash
npm install
```

2. Install Python dependencies:
```bash
pip install fastapi uvicorn pandas pyyaml
```

### Running the Dashboard

1. **Start the FastAPI backend** (in one terminal):
```bash
cd api
uvicorn app:app --reload --port 8000
```

2. **Start the Next.js frontend** (in another terminal):
```bash
npm run dev
```

3. **Open your browser** to `http://localhost:3000`

## API Endpoints

The dashboard connects to these FastAPI endpoints:

- `GET /api/metrics` - Model performance metrics
- `GET /api/predictions` - Adjuvant predictions with filtering
- `GET /api/validation` - Computational validation results
- `GET /api/kg/stats` - Knowledge graph statistics
- `GET /api/kg/sample` - Sample graph data for visualization

## Configuration

Set the API base URL via environment variable:
```bash
export NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Development

### Adding New Components

1. Create components in `components/ui/` following shadcn/ui patterns
2. Use the `cn()` utility for class name merging
3. Follow the existing component structure and styling

### Styling

- Use Tailwind CSS classes for styling
- Follow the design system defined in `tailwind.config.js`
- Use CSS variables for consistent theming

### Data Fetching

- All data fetching is done in the main dashboard component
- Use the `useEffect` hook for initial data loading
- Implement proper loading states and error handling

## Building for Production

```bash
npm run build
npm start
```

## Troubleshooting

### Common Issues

1. **API Connection Failed**: Ensure FastAPI backend is running on port 8000
2. **Graph Not Loading**: Check that pipeline results exist in `../results/`
3. **Styling Issues**: Verify Tailwind CSS is properly configured

### Debug Mode

Enable debug logging in the browser console to see API requests and responses.

## Contributing

1. Follow the existing code structure and patterns
2. Use TypeScript for type safety
3. Test components in isolation
4. Ensure responsive design works on different screen sizes

## License

This dashboard is part of the β-lactam adjuvant discovery pipeline project.
