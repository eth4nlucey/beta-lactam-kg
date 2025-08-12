'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import dynamic from 'next/dynamic';

// Dynamically import Cytoscape to avoid SSR issues
const CytoscapeComponent = dynamic(() => import('react-cytoscapejs'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96">Loading graph...</div>
});

interface Metrics {
  mrr: number;
  hits_at_1: number;
  hits_at_3: number;
  hits_at_10: number;
  auroc: number;
  epochs: number;
  embedding_dim: number;
  num_entities: number;
  num_relations: number;
  train_size: number;
  valid_size: number;
  test_size: number;
}

interface Prediction {
  drug: string;
  adjuvant: string;
  relation: string;
  score: number;
}

interface Validation {
  drug_a: string;
  drug_b: string;
  model_score: number;
  epmc_hits: number;
  top_pmids: string[];
  drugcomb_found: boolean;
  synergy_metric: string;
  synergy_value: number | null;
}

interface KGStats {
  total_entities: number;
  total_edges: number;
  total_relations: number;
  entity_types: Record<string, number>;
  relation_counts: Record<string, number>;
  data_sources: Record<string, number>;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [validations, setValidations] = useState<Validation[]>([]);
  const [kgStats, setKgStats] = useState<KGStats | null>(null);
  const [graphData, setGraphData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [drugFilter, setDrugFilter] = useState('');

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch all data in parallel
      const [metricsRes, predictionsRes, validationsRes, kgStatsRes, graphRes] = await Promise.all([
        fetch(`${API_BASE}/api/metrics`),
        fetch(`${API_BASE}/api/predictions?limit=100`),
        fetch(`${API_BASE}/api/validation?limit=100`),
        fetch(`${API_BASE}/api/kg/stats`),
        fetch(`${API_BASE}/api/kg/sample?n=200`)
      ]);

      if (metricsRes.ok) setMetrics(await metricsRes.json());
      if (predictionsRes.ok) setPredictions((await predictionsRes.json()).predictions);
      if (validationsRes.ok) setValidations((await validationsRes.json()).validations);
      if (kgStatsRes.ok) setKgStats(await kgStatsRes.json());
      if (graphRes.ok) setGraphData(await graphRes.json());
      
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredPredictions = predictions.filter(p => 
    p.drug.toLowerCase().includes(drugFilter.toLowerCase()) ||
    p.adjuvant.toLowerCase().includes(drugFilter.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          β-Lactam Adjuvant Discovery Dashboard
        </h1>

        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Entities</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{kgStats?.total_entities.toLocaleString() || 0}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Edges</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{kgStats?.total_edges.toLocaleString() || 0}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Model MRR</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(metrics?.mrr || 0).toFixed(4)}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Model AUROC</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(metrics?.auroc || 0).toFixed(4)}</div>
            </CardContent>
          </Card>
        </div>

        {/* Model Performance Chart */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { metric: 'Hits@1', value: metrics?.hits_at_1 || 0 },
                { metric: 'Hits@3', value: metrics?.hits_at_3 || 0 },
                { metric: 'Hits@10', value: metrics?.hits_at_10 || 0 },
                { metric: 'MRR', value: (metrics?.mrr || 0) * 100 },
                { metric: 'AUROC', value: (metrics?.auroc || 0) * 100 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Predictions Table */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Top Adjuvant Predictions</CardTitle>
            <div className="flex items-center space-x-2">
              <Input
                placeholder="Filter by drug or adjuvant..."
                value={drugFilter}
                onChange={(e) => setDrugFilter(e.target.value)}
                className="max-w-sm"
              />
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Drug</TableHead>
                  <TableHead>Adjuvant</TableHead>
                  <TableHead>Relation</TableHead>
                  <TableHead>Score</TableHead>
                  <TableHead>Validation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredPredictions.slice(0, 20).map((pred, index) => {
                  const validation = validations.find(v => 
                    v.drug_a === pred.drug && v.drug_b === pred.adjuvant
                  );
                  
                  return (
                    <TableRow key={index}>
                      <TableCell className="font-medium">{pred.drug}</TableCell>
                      <TableCell>{pred.adjuvant}</TableCell>
                      <TableCell>
                        <Badge variant="secondary">{pred.relation}</Badge>
                      </TableCell>
                      <TableCell>{pred.score.toFixed(4)}</TableCell>
                      <TableCell>
                        {validation ? (
                          <div className="space-y-1">
                            <Badge variant={validation.epmc_hits > 0 ? "default" : "secondary"}>
                              {validation.epmc_hits} PMIDs
                            </Badge>
                            {validation.drugcomb_found && (
                              <Badge variant="outline">DrugComb</Badge>
                            )}
                          </div>
                        ) : (
                          <Badge variant="outline">No validation</Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Knowledge Graph Visualization */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Knowledge Graph Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            {graphData ? (
              <div className="h-96 border rounded-lg">
                <CytoscapeComponent
                  elements={graphData}
                  style={{ width: '100%', height: '100%' }}
                  cy={(cy) => {
                    cy.layout({
                      name: 'cose',
                      animate: 'end',
                      animationDuration: 1000
                    }).run();
                  }}
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 text-gray-500">
                Graph data not available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Data Source Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Data Source Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={Object.entries(kgStats?.data_sources || {}).map(([source, count]) => ({
                source,
                count
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
