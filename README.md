# FOMC RAG System

> An AI-powered Retrieval-Augmented Generation system for analyzing Federal Open Market Committee (FOMC) documents using semantic search, vector embeddings, and a ReAct agent pattern.

---

## Table of Contents

1. [Assessment Submission Overview](#assessment-submission-overview)
2. [System Architecture](#system-architecture)
3. [Development Environment Setup](#development-environment-setup)
4. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
5. [Engineering Deep Dive](#engineering-deep-dive)
6. [Production Considerations](#production-considerations)

---

## Assessment Submission Overview

### Project Deliverables

| Component | Description | Location |
|-----------|-------------|----------|
| **MCP Server** | Model Context Protocol server with 12 FOMC tools | `code/src/FOMC.McpServer/` |
| **ReAct Agent** | Autonomous agent with dynamic tool selection | `code/src/FOMC.Agent/` |
| **Vector Store** | Qdrant with HNSW indexing | `qdrant/` |
| **Shared Library** | Common abstractions and utilities | `code/src/FOMC.Shared/` |

### Key Technical Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Vector Index** | HNSW (M=16, EfConstruct=100) | Optimal for 1000s of documents; O(log n) search |
| **Embedding Model** | text-embedding-3-large (3072 dims) | Best quality for financial domain |
| **LLM** | gpt-5-chat via Azure OpenAI | Advanced chat model for ReAct agents |
| **Chunking Strategy** | Adaptive (400/700 chars by doc type) | Preserves context for different formats |
| **Agent Pattern** | ReAct (Reasoning + Acting) | Transparent, auditable decision chain |

### Why HNSW Over IVF?

For a corpus of ~200-5000 documents:

| Aspect | HNSW | IVF |
|--------|------|-----|
| **Index Build** | Slower (graph construction) | Faster (k-means) |
| **Memory** | Higher (graph structure) | Lower (centroids only) |
| **Accuracy** | Higher (navigable small world) | Lower (cluster boundary issues) |
| **Best For** | <100K documents | Millions of documents |

**Verdict**: HNSW is ideal for this scale. IVF would be overkill and introduce cluster boundary accuracy issues.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FOMC RAG System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐  │
│  │   User      │───▶│     ReAct Agent         │───▶│    MCP Server       │  │
│  │   Query     │    │   (FOMC.Agent)          │    │  (FOMC.McpServer)   │  │
│  └─────────────┘    │                         │    │                     │  │
│                     │  1. Thought: Analyze    │    │  12 Specialized     │  │
│                     │  2. Action: Select Tool │───▶│  FOMC Tools         │  │
│                     │  3. Observation: Result │◀───│                     │  │
│                     │  4. Repeat or Answer    │    └──────────┬──────────┘  │
│                     └─────────────────────────┘               │             │
│                                                               ▼             │
│                     ┌─────────────────────────────────────────────────────┐ │
│                     │                   Qdrant                            │ │
│                     │         Vector Database (HNSW Index)                │ │
│                     │  ┌─────────────────────────────────────────────┐    │ │
│                     │  │ Collection: fomc_documents                 │    │ │
│                     │  │ • 3072-dim embeddings (text-embedding-3-lg)│    │ │
│                     │  │ • Payload: meeting_date, doc_type, text    │    │ │
│                     │  │ • HNSW: M=16, EfConstruct=100              │    │ │
│                     │  └─────────────────────────────────────────────┘    │ │
│                     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ReAct Agent Flow

```
User Query: "What did the Fed say about inflation in January 2024?"
                │
                ▼
        ┌───────────────┐
        │    THOUGHT    │  "I need to search for inflation discussions
        │               │   in January 2024 FOMC documents"
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │    ACTION     │  Tool: search_fomc_documents
        │               │  Args: {query: "inflation", date: "2024-01"}
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  OBSERVATION  │  [Retrieved chunks with scores 0.89, 0.85, 0.82]
        │               │  "Core PCE inflation remained elevated at 2.9%..."
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │    THOUGHT    │  "I have sufficient context. The chunks show
        │               │   specific inflation figures and Fed stance."
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │    ANSWER     │  "In January 2024, the Fed noted that core PCE
        │               │   inflation was at 2.9%, above the 2% target..."
        └───────────────┘
```

---

## Development Environment Setup

### Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| .NET SDK | 10.0+ | Build and run the solution |
| Qdrant | 1.7+ | Vector database |
| Azure OpenAI | - | Embeddings and LLM |
| Python | 3.10+ | Benchmark scripts (optional) |
| VS Code | Latest | Recommended IDE |

### Step 1: Clone and Configure

```powershell
# Clone the repository
git clone <repository-url>
cd FOMC

# Copy example configuration files
Copy-Item code\src\FOMC.Agent\appsettings.example.json code\src\FOMC.Agent\appsettings.json
Copy-Item code\src\FOMC.McpServer\appsettings.example.json code\src\FOMC.McpServer\appsettings.json
```

### Step 2: Configure Azure OpenAI

Edit both `appsettings.json` files:

```json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com/",
    "ApiKey": "your-api-key",
    "ChatDeployment": "gpt-5-chat",
    "EmbeddingDeployment": "text-embedding-3-large"
  }
}
```

### Step 3: Start Qdrant

```powershell
# Option A: Run included binary (Windows)
cd qdrant\qdrant-x86_64-pc-windows-msvc
.\qdrant.exe --config-path ..\config\qdrant_config.yaml

# Option B: Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Step 4: Build and Run

```powershell
# Build entire solution
dotnet build code\FOMC.slnx

# Run the MCP Server
dotnet run --project code\src\FOMC.McpServer\FOMC.McpServer.csproj

# In another terminal, run the Agent
dotnet run --project code\src\FOMC.Agent\FOMC.Agent.csproj
```

### Step 5: Python Benchmarks (Optional)

```powershell
# Install Python dependencies
pip install -r scripts\requirements.txt

# Run benchmarks
python scripts\benchmark.py
```

### Project Structure

```
FOMC/
├── code/
│   ├── FOMC.slnx                    # Solution file
│   └── src/
│       ├── FOMC.Agent/              # ReAct agent implementation
│       │   ├── FomcAgent.cs         # Main agent with ReAct loop
│       │   ├── RagEvaluator.cs      # Retrieval evaluation
│       │   └── data/raw/            # FOMC HTML documents
│       ├── FOMC.McpServer/          # MCP server with tools
│       │   ├── Program.cs           # Server entry point
│       │   └── Tools/
│       │       └── FomcSearchTools.cs  # 12 FOMC tools
│       └── FOMC.Shared/             # Shared library
│           ├── Abstractions/        # Interfaces
│           ├── Configuration/       # Options classes
│           ├── Constants/           # Defaults, PayloadFields
│           ├── Models/              # FomcChunk, SearchResult
│           └── Utilities/           # Helpers
├── qdrant/                          # Vector database
│   ├── config/                      # Qdrant configuration
│   └── qdrant-data/                 # Persisted data
├── scripts/
│   ├── benchmark.py                 # Performance benchmarks
│   └── requirements.txt             # Python dependencies
└── README.md                        # This file
```

---

## Design Decisions & Trade-offs

### 1. Chunking Strategy

Different document types require different chunking approaches:

| Document Type | Chunk Size | Overlap | Rationale |
|--------------|------------|---------|-----------|
| **Press Statements** | 400 chars | 60 chars | Dense, concise policy announcements |
| **Meeting Minutes** | 700 chars | 100 chars | Longer discussions need more context |

**Implementation**: `DocumentChunker.cs` applies adaptive chunking based on `doc_type` payload field.

**Production Consideration**: For millions of filings, implement hierarchical chunking with parent-child relationships. Store summary chunks for broad queries and detail chunks for specific lookups. Use document structure (headings, sections) for semantic boundaries.

### 2. Embedding Model Selection

| Model | Dimensions | Quality | Cost |
|-------|-----------|---------|------|
| text-embedding-3-small | 1536 | Good | Low |
| **text-embedding-3-large** | 3072 | Best | Medium |
| ada-002 | 1536 | Legacy | Low |

**Choice**: text-embedding-3-large provides superior semantic understanding for financial terminology at reasonable cost.

**Production Consideration**: For huge document volumes, consider dimensionality reduction via Matryoshka embeddings (text-embedding-3-large supports 256, 1024, or 3072 dims). Use 1024 dims for 60% storage savings with ~2% quality loss.

### 3. Confidence Thresholds

```csharp
public static class Defaults
{
    public const double HighConfidenceThreshold = 0.85;
    public const double MediumConfidenceThreshold = 0.75;
    public const double LowConfidenceThreshold = 0.65;
}
```

| Threshold | Score Range | Agent Behavior |
|-----------|-------------|----------------|
| High | ≥ 0.85 | Directly cite as authoritative |
| Medium | 0.75 - 0.84 | Use with caveats |
| Low | 0.65 - 0.74 | Request clarification or additional search |
| Below | < 0.65 | Do not use |

**Production Consideration**: Implement dynamic threshold adjustment based on query type. Factual queries need higher thresholds; exploratory queries can use lower thresholds. Log threshold decisions for A/B testing.

### 4. HNSW Parameters

```csharp
public const int HnswM = 16;           // Connections per node
public const int HnswEfConstruct = 100; // Construction quality
public const int SearchEf = 64;         // Search quality
```

| Parameter | Value | Trade-off |
|-----------|-------|-----------|
| **M** | 16 | Memory ↔ Recall (16 is balanced) |
| **EfConstruct** | 100 | Build time ↔ Index quality |
| **SearchEf** | 64 | Search speed ↔ Accuracy |

**Production Consideration**: For millions of documents, increase M to 32-64 and EfConstruct to 200. Use scalar quantization to reduce memory by 75%. Implement sharding across multiple Qdrant nodes.

---

## Engineering Deep Dive

### MCP Tools Reference

The MCP Server exposes 12 specialized tools for FOMC document analysis:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `search_fomc_documents` | Semantic search across all documents | query, top_k, min_score |
| `search_by_date_range` | Filter by meeting date range | start_date, end_date, query |
| `search_by_document_type` | Filter by document type | doc_type, query |
| `get_document_by_id` | Retrieve specific document | document_id |
| `list_fomc_meetings` | List all available meetings | year (optional) |
| `get_meeting_summary` | Get meeting overview | meeting_date |
| `compare_meetings` | Compare two meetings | date1, date2, aspect |
| `search_policy_changes` | Find policy rate changes | year (optional) |
| `search_economic_indicators` | Search for indicator mentions | indicator, date_range |
| `get_fed_funds_rate_history` | Rate decisions over time | start_date, end_date |
| `search_voting_records` | Find voting patterns | member_name (optional) |
| `get_document_statistics` | Corpus statistics | none |

### Anti-Hallucination Safeguards

The system prompt includes strict grounding rules:

```
§3 — ANTI-HALLUCINATION RULES
────────────────────────────
CRITICAL: Never invent or assume information.

DATE-RANGE VALIDATION:
1. ALWAYS call list_fomc_meetings FIRST for any date-bounded query
2. If user asks about years outside corpus → state "No documents available for [years]"
3. The document's meeting_date field is authoritative, NOT dates mentioned in text
4. A 2024 document discussing 2020 events is a 2024 source, not a 2020 source

GROUNDING REQUIREMENTS:
• Only cite information from retrieved chunks
• Include confidence scores in citations
• If chunks are insufficient, say "I don't have enough information"
```

### Token Usage Tracking

```csharp
public interface ITokenTracker
{
    void TrackUsage(string operation, int inputTokens, int outputTokens);
    TokenUsageSnapshot GetSnapshot();
    decimal CalculateCost(PricingOptions pricing);
}
```

All API calls are tracked for cost monitoring and optimization.

---

## Production Considerations

### Scaling to Millions of Filings

For enterprise deployments at firms like Janus Henderson with huge document volumes:

#### 1. Distributed Vector Storage

```yaml
# Qdrant cluster configuration
cluster:
  enabled: true
  shards: 10
  replicas: 2
  
collections:
  fomc_documents:
    shard_number: 10
    replication_factor: 2
    on_disk_payload: true  # Reduce memory
```

#### 2. Tiered Storage

| Tier | Content | Storage | Access Time |
|------|---------|---------|-------------|
| Hot | Last 2 years | In-memory | <10ms |
| Warm | 2-10 years | SSD | <50ms |
| Cold | >10 years | Object storage | <500ms |

#### 3. Query Optimization

- **Query caching**: Cache frequent queries (Fed funds rate history, latest meeting)
- **Batch embeddings**: Process new documents in batches during off-hours
- **Async ingestion**: Queue-based document processing with backpressure

#### 4. Index Partitioning

```csharp
// Partition by year for efficient date-range queries
var collections = new[]
{
    "fomc_2024", "fomc_2023", "fomc_2022", // ...
};

// Query only relevant partitions
var relevantCollections = GetCollectionsForDateRange(startDate, endDate);
```