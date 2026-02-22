using FOMC.Agent;
using FOMC.Agent.Benchmark;
using FOMC.Data.Factories;
using FOMC.Data.Ingestion;
using FOMC.Data.Services;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Configuration;
using FOMC.Shared.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// ── FOMC Agentic RAG — CLI Demo ───────────────────────────────────────────────
//
// Commands:
//   (default)  Interactive Q&A agent
//   ingest     Download, chunk, embed, and index all FOMC documents
//   benchmark  Run HNSW ef-search recall/latency sweep
//   help       Show help
// ─────────────────────────────────────────────────────────────────────────────

var mode = args.FirstOrDefault()?.ToLowerInvariant() ?? "chat";

// Build host for DI (used by ingest and benchmark modes)
var host = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration(cfg =>
    {
        cfg.AddJsonFile("appsettings.json", optional: false);
        cfg.AddEnvironmentVariables("FOMC_");
    })
    .ConfigureLogging((ctx, log) =>
    {
        log.ClearProviders();
        log.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
        // Log levels controlled by appsettings.json Logging:LogLevel section
        log.AddConfiguration(ctx.Configuration.GetSection("Logging"));
    })
    .ConfigureServices((ctx, svc) =>
    {
        svc.AddHttpClient();
        
        // Configuration options (strongly-typed)
        svc.Configure<AzureOpenAIOptions>(ctx.Configuration.GetSection(AzureOpenAIOptions.SectionName));
        svc.Configure<QdrantOptions>(ctx.Configuration.GetSection(QdrantOptions.SectionName));
        svc.Configure<PricingOptions>(ctx.Configuration.GetSection(PricingOptions.SectionName));
        svc.Configure<ChunkEnrichmentOptions>(ctx.Configuration.GetSection(ChunkEnrichmentOptions.SectionName));
        
        // Factories
        svc.AddSingleton<IAzureOpenAIClientFactory, AzureOpenAIClientFactory>();
        
        // Core services
        svc.AddSingleton<TokenTracker>();
        svc.AddSingleton<ITokenTracker>(sp => sp.GetRequiredService<TokenTracker>());
        svc.AddSingleton<ICostCalculator, CostCalculator>();
        svc.AddSingleton<EmbeddingService>();
        svc.AddSingleton<IEmbeddingService>(sp => sp.GetRequiredService<EmbeddingService>());
        
        // Vocabulary service for sparse vectors (BM25 in Qdrant)
        svc.AddSingleton<VocabularyService>(sp =>
        {
            var config = sp.GetRequiredService<IConfiguration>();
            var vocabPath = config["Vocabulary:Path"] ?? "data/vocabulary.json";
            var logger = sp.GetRequiredService<ILogger<VocabularyService>>();
            return new VocabularyService(logger, vocabPath);
        });
        
        svc.AddSingleton<QdrantService>();
        svc.AddSingleton<IVectorStore>(sp => sp.GetRequiredService<QdrantService>());
        svc.AddSingleton<FomcDownloader>(sp =>
        {
            var http = sp.GetRequiredService<IHttpClientFactory>().CreateClient();
            var config = sp.GetRequiredService<IConfiguration>();
            var cacheDir = config["DataCache:Directory"] ?? "data/raw";
            var logger = sp.GetRequiredService<ILogger<FomcDownloader>>();
            http.DefaultRequestHeaders.UserAgent.ParseAdd("FOMC-RAG/1.0 (research prototype)");
            http.Timeout = TimeSpan.FromSeconds(60);
            return new FomcDownloader(http, cacheDir, logger);
        });
        svc.AddSingleton<ChunkEnricher>();
        svc.AddSingleton<IChunkEnricher>(sp => sp.GetRequiredService<ChunkEnricher>());
        svc.AddTransient<IngestionPipeline>();
        svc.AddSingleton<IndexBenchmark>();
    })
    .Build();

switch (mode)
{
    case "ingest":
        await RunIngestion(host);
        break;

    case "benchmark":
        await RunBenchmark(host);
        break;

    case "eval":
        await RunEval(host);
        break;

    case "help":
        PrintHelp();
        break;

    default:
        await RunChat(host);
        break;
}

// ── Mode handlers ─────────────────────────────────────────────────────────────

static async Task RunIngestion(IHost host)
{
    Console.WriteLine("\n=== FOMC Ingestion Pipeline ===\n");
    var pipeline = host.Services.GetRequiredService<IngestionPipeline>();
    await pipeline.RunAsync();
    Console.WriteLine("\nIngestion complete. Run without arguments to start the agent.");
}

static async Task RunBenchmark(IHost host)
{
    Console.WriteLine("\n=== HNSW Index Benchmark ===\n");
    var bench = host.Services.GetRequiredService<IndexBenchmark>();
    await bench.RunAsync();
}

static async Task RunEval(IHost host)
{
    Console.WriteLine("\n=== RAG Evaluation ===\n");
    var config  = host.Services.GetRequiredService<IConfiguration>();
    var tokens  = host.Services.GetRequiredService<TokenTracker>();
    var costCalc = host.Services.GetRequiredService<ICostCalculator>();
    var logger  = host.Services.GetRequiredService<ILogger<FomcAgent>>();
    var evalLog = host.Services.GetRequiredService<ILogger<RagEvaluator>>();

    await using var agent = await FomcAgent.CreateAsync(config, tokens, logger);
    var evaluator = new RagEvaluator(config, tokens, evalLog);
    await evaluator.RunFullEvalAsync(agent);
    PrintTokenReport(tokens, costCalc);
}

static async Task RunChat(IHost host)
{
    var config  = host.Services.GetRequiredService<IConfiguration>();
    var tokens  = host.Services.GetRequiredService<TokenTracker>();
    var costCalc = host.Services.GetRequiredService<ICostCalculator>();
    var logger  = host.Services.GetRequiredService<ILogger<FomcAgent>>();
    var vocab   = host.Services.GetRequiredService<VocabularyService>();
    
    // Load vocabulary for hybrid search
    if (await vocab.LoadAsync())
    {
        var stats = vocab.GetStats();
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"[Vocabulary loaded: {stats.TermCount:N0} terms, {stats.DocCount:N0} docs — hybrid search ready]");
        Console.ResetColor();
    }
    else
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("[Note: No vocabulary found. Run 'ingest' to enable hybrid search.]");
        Console.ResetColor();
    }

    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("""
        ╔═══════════════════════════════════════════════════════════════╗
        ║            FOMC Research Assistant  (Agentic RAG)            ║
        ║  Powered by Qdrant + Azure AI Foundry + Model Context Protocol║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  🔄 Multi-turn chat history enabled                          ║
        ║  📚 Contextual Retrieval + Hybrid BM25 search                ║
        ╚═══════════════════════════════════════════════════════════════╝
        Commands:
          Type your question and press Enter
          'clear'  - Clear chat history and start fresh
          'exit'   - Quit the assistant
        """);
    Console.ResetColor();

    await using var agent = await FomcAgent.CreateAsync(config, tokens, logger);

    // Sample questions demonstrating advanced RAG capabilities for asset management use cases
    var sampleQuestions = new[]
    {
        // Multi-hop reasoning: requires synthesizing across multiple meetings
        "Compare the Fed's inflation outlook between Q1 2024 and Q1 2025 — what changed?",
        
        // Temporal analysis: tracking policy evolution
        "Trace the evolution of the Fed's stance on rate cuts from January 2024 to September 2025.",
        
        // Risk analysis: relevant for portfolio positioning
        "What downside risks to GDP growth did the FOMC identify in 2025, and how might they affect equity allocations?",
        
        // Policy decision extraction: direct impact on fixed income
        "Summarize all federal funds rate decisions in 2024-2025 with the key rationale for each.",
        
        // Narrative analysis: understanding Fed communication shifts  
        "How has the Fed's language around 'data dependency' evolved, and what does this signal for forward guidance?",
        
        // Cross-document synthesis: stress testing scenarios
        "What inflation scenarios did the FOMC consider, and which indicators would trigger a policy reversal?",
        
        // Balance sheet implications: QT impact on liquidity
        "What did the Fed say about balance sheet reduction (QT) and its impact on financial conditions?"
    };

    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine("\nSample questions:");
    foreach (var q in sampleQuestions)
        Console.WriteLine($"  • {q}");
    Console.ResetColor();
    Console.WriteLine();

    // Chat history for multi-turn conversations
    var chatHistory = new List<AgentMessage>();
    const int MaxHistoryTurns = 10; // Keep last N turns to manage context window

    while (true)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write("You: ");
        Console.ResetColor();

        var input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input)) continue;
        if (input.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;
        
        // Clear history command
        if (input.Equals("clear", StringComparison.OrdinalIgnoreCase))
        {
            chatHistory.Clear();
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("Chat history cleared.\n");
            Console.ResetColor();
            continue;
        }

        Console.ForegroundColor = ConsoleColor.DarkCyan;
        Console.WriteLine("\nResearching... (calling MCP tools)\n");
        Console.ResetColor();

        try
        {
            // Pass chat history for context-aware responses
            var response = await agent.AskAsync(input, chatHistory);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("Agent: ");
            Console.ResetColor();
            Console.WriteLine(response.Answer);

            if (response.Citations.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"\n[{response.ToolCallsUsed} tool call(s), {response.Citations.Count} source(s), {chatHistory.Count/2 + 1} turn(s)]");
                Console.ResetColor();
            }

            if (response.Uncertainty is not null)
            {
                Console.ForegroundColor = ConsoleColor.DarkYellow;
                Console.WriteLine($"Note: {response.Uncertainty}");
                Console.ResetColor();
            }

            // Add this exchange to history for multi-turn context
            chatHistory.Add(new AgentMessage { Role = "user", Content = input });
            chatHistory.Add(new AgentMessage { Role = "assistant", Content = response.Answer });

            // Trim history to prevent context window overflow
            while (chatHistory.Count > MaxHistoryTurns * 2)
            {
                chatHistory.RemoveAt(0);
                chatHistory.RemoveAt(0);
            }
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Error: {ex.Message}");
            Console.ResetColor();
        }

        Console.WriteLine();
    }

    PrintTokenReport(tokens, costCalc);
}

static void PrintTokenReport(TokenTracker tokens, ICostCalculator costCalc)
{
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine(costCalc.FormatCostReport(tokens.Snapshot()));
    Console.ResetColor();
}

static void PrintHelp()
{
    Console.WriteLine("""
        FOMC Agentic RAG — CLI

        Usage:
          dotnet run                 Start interactive Q&A agent
          dotnet run -- ingest       Download and index FOMC documents
          dotnet run -- benchmark    Run HNSW ef-search benchmark
          dotnet run -- help         Show this help

        Configuration:
          Edit appsettings.json to set AzureOpenAI:Endpoint, ApiKey,
          EmbeddingDeployment, and ChatDeployment.
          Set McpServer:ExecutablePath to the FOMC.McpServer binary path.

        dotnet run -- eval         Run RAG accuracy evaluation (10 test questions)
        """);
}
