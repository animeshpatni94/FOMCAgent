using FOMC.Data.Factories;
using FOMC.Data.Services;
using FOMC.McpServer.Tools;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Configuration;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// ── FOMC RAG — MCP Server ────────────────────────────────────────────────────
// Runs as a subprocess communicating over stdin/stdout (stdio transport).
//
// The agent (FOMC.Agent) spawns this process and invokes its tools via the
// Model Context Protocol.  All retrieval passes through here — the agent
// never queries Qdrant directly, enforcing clean tool-use grounding.
//
// MCP TOOLS EXPOSED:
//   - search_fomc_documents, search_with_rerank, search_with_hyde
//   - search_with_expansion, search_decomposed, search_with_context
//   - search_key_decisions, list_fomc_meetings, get_chunk_context
//   - evaluate_retrieval (CRAG), search_with_critique (Self-Reflective RAG)
// ─────────────────────────────────────────────────────────────────────────────

var builder = Host.CreateApplicationBuilder(args);

// FIX: Set base path to executable directory so appsettings.json is found
// when spawned as subprocess from a different working directory
var exeDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)!;
builder.Configuration.SetBasePath(exeDir);

// Configuration: appsettings.json → environment variables
builder.Configuration
    .AddJsonFile("appsettings.json", optional: false)
    .AddEnvironmentVariables("FOMC_");

// CRITICAL: suppress console logging — stdout is used exclusively for MCP protocol.
// Redirect logs to stderr so they don't corrupt the stdio stream.
// Log levels are controlled by appsettings.json Logging:LogLevel section.
builder.Logging
    .ClearProviders()
    .AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);

// ── Configuration Options (strongly-typed) ──────────────────────────────────
builder.Services.Configure<AzureOpenAIOptions>(
    builder.Configuration.GetSection(AzureOpenAIOptions.SectionName));
builder.Services.Configure<QdrantOptions>(
    builder.Configuration.GetSection(QdrantOptions.SectionName));
builder.Services.Configure<QueryEnhancementOptions>(
    builder.Configuration.GetSection(QueryEnhancementOptions.SectionName));
builder.Services.Configure<RerankingOptions>(
    builder.Configuration.GetSection(RerankingOptions.SectionName));

// ── Factories ────────────────────────────────────────────────────────────────
builder.Services.AddSingleton<IAzureOpenAIClientFactory, AzureOpenAIClientFactory>();

// ── Data Services (interfaces for DIP compliance) ───────────────────────────
builder.Services.AddHttpClient();
builder.Services.AddSingleton<ITokenTracker, TokenTracker>();
builder.Services.AddSingleton<IEmbeddingService, EmbeddingService>();

// VocabularyService is required by QdrantService for sparse vector (BM25) search
builder.Services.AddSingleton<VocabularyService>(sp =>
{
    var vocabPath = Path.Combine(exeDir, "..", "..", "..", "..", "FOMC.Agent", "data", "vocabulary.json");
    vocabPath = Path.GetFullPath(vocabPath); // Normalize path
    var logger = sp.GetRequiredService<ILogger<VocabularyService>>();
    return new VocabularyService(logger, vocabPath);
});

// Vector store with decorator pattern: QdrantService → KeyPassageBoostingVectorStore
// This applies soft score boosting to key passages (policy decisions) across ALL search tools,
// reducing bias from hard filtering while prioritizing important content.
builder.Services.AddSingleton<QdrantService>();
builder.Services.AddSingleton<IVectorStore>(sp => 
    new KeyPassageBoostingVectorStore(sp.GetRequiredService<QdrantService>()));

builder.Services.AddSingleton<IQueryEnhancer, QueryEnhancer>();
builder.Services.AddSingleton<IReranker, Reranker>();
builder.Services.AddSingleton<IRetrievalEvaluator, RetrievalEvaluator>();

// Register MCP server with stdio transport and tools
builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithTools<FomcSearchTools>();

var host = builder.Build();

// DEBUG: Wait for debugger to attach when WAIT_FOR_DEBUGGER env var is set
// Usage: Set FOMC_WAIT_FOR_DEBUGGER=true, then use "Attach to MCP Server" in VS Code
if (Environment.GetEnvironmentVariable("FOMC_WAIT_FOR_DEBUGGER") == "true")
{
    Console.Error.WriteLine($"[MCP Server] Waiting for debugger... PID={Environment.ProcessId}");
    while (!System.Diagnostics.Debugger.IsAttached)
    {
        await Task.Delay(500);
    }
    Console.Error.WriteLine("[MCP Server] Debugger attached!");
}

await host.RunAsync();
