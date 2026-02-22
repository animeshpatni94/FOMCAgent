using System.ClientModel;
using System.Text.Json;
using System.Text.RegularExpressions;
using FOMC.Data.Factories;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Configuration;
using FOMC.Shared.Models;
using FOMC.Shared.Utilities;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenAI.Chat;

namespace FOMC.Data.Ingestion;

/// <summary>
/// Creative ingestion enhancement: enriches chunks with LLM-generated summaries,
/// extracted entities, and key passage detection.
/// Implements IChunkEnricher for testability.
/// </summary>
public sealed partial class ChunkEnricher : IChunkEnricher
{
    private readonly ChatClient _chat;
    private readonly ChunkEnrichmentOptions _options;
    private readonly ILogger<ChunkEnricher> _logger;
    private readonly bool _enabled;

    private const int MaxConcurrentCalls = 2;  // Reduced to avoid rate limits
    private const int MaxRetries = 5;
    private static readonly TimeSpan RetryDelay = TimeSpan.FromSeconds(60);

    public ChunkEnricher(
        IAzureOpenAIClientFactory clientFactory,
        IOptions<ChunkEnrichmentOptions> options,
        ILogger<ChunkEnricher> logger)
    {
        _logger = logger;
        _options = options.Value;
        _enabled = _options.Enabled;

        if (!_enabled)
        {
            _logger.LogInformation("Chunk enrichment disabled.");
            _chat = null!;
            return;
        }

        _chat = clientFactory.CreateChatClient();
        _logger.LogInformation("ChunkEnricher initialized with model: {Model}", clientFactory.ChatModel);
    }

    /// <summary>
    /// Enrich a batch of chunks with summaries, entities, and key passage flags.
    /// </summary>
    public async Task<List<FomcChunk>> EnrichAsync(
        List<FomcChunk> chunks,
        string? documentSummary = null,
        CancellationToken ct = default)
    {
        if (!_enabled || chunks.Count == 0)
        {
            _logger.LogInformation("Chunk enrichment skipped (enabled={Enabled}, count={Count})", _enabled, chunks.Count);
            return chunks;
        }

        _logger.LogInformation("Enriching {Count} chunks with LLM (max {Concurrent} concurrent)...", 
            chunks.Count, MaxConcurrentCalls);
        var semaphore = new SemaphoreSlim(MaxConcurrentCalls);
        int completed = 0;

        var tasks = chunks.Select(async chunk =>
        {
            await semaphore.WaitAsync(ct);
            try
            {
                var result = await EnrichChunkAsync(chunk, documentSummary, ct);
                var done = Interlocked.Increment(ref completed);
                if (done % 10 == 0 || done == chunks.Count)
                    _logger.LogInformation("  Enrichment progress: {Done}/{Total}", done, chunks.Count);
                return result;
            }
            finally
            {
                semaphore.Release();
            }
        });

        var results = await Task.WhenAll(tasks);
        _logger.LogInformation("Chunk enrichment complete: {Count} chunks processed", results.Length);
        return results.ToList();
    }

    /// <summary>
    /// Generate a document-level summary from the first few chunks.
    /// This summary is shared across all chunks for context.
    /// </summary>
    public async Task<string?> GenerateDocumentSummaryAsync(
        List<FomcChunk> chunks,
        CancellationToken ct = default)
    {
        if (!_enabled || chunks.Count == 0)
        {
            _logger.LogDebug("Document summary skipped (enabled={Enabled})", _enabled);
            return null;
        }

        _logger.LogDebug("Generating document summary from first {Count} chunks...", Math.Min(3, chunks.Count));

        // Take first 3 chunks (usually contains the key decisions)
        var sample = string.Join("\n\n", chunks.Take(3).Select(c => c.Text));

        var messages = new List<ChatMessage>
        {
            ChatMessage.CreateSystemMessage(
                "You are a financial analyst summarizing FOMC documents. " +
                "Write a 2-3 sentence executive summary of the key policy decisions and economic outlook."),
            ChatMessage.CreateUserMessage(sample)
        };

        try
        {
            var options = new ChatCompletionOptions { MaxOutputTokenCount = 150 };
            _logger.LogDebug("Calling LLM for document summary...");
            var response = await _chat.CompleteChatAsync(messages, options, ct);
            var summary = response.Value.Content[0].Text?.Trim();
            _logger.LogDebug("Document summary generated ({Len} chars)", summary?.Length ?? 0);
            return summary;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to generate document summary: {Message}", ex.Message);
            return null;
        }
    }

    private async Task<FomcChunk> EnrichChunkAsync(
        FomcChunk chunk,
        string? documentSummary,
        CancellationToken ct)
    {
        // No fallback - always use LLM enrichment with retries
        var (summary, entities, isKey) = await AnalyzeChunkAsync(chunk.Text, ct);

        return chunk with
        {
            Summary         = summary,
            KeyEntities     = entities,
            DocumentSummary = documentSummary,
            IsKeyPassage    = isKey
        };
    }

    private async Task<(string? Summary, List<string> Entities, bool IsKey)> AnalyzeChunkAsync(
        string text,
        CancellationToken ct)
    {
        var messages = new List<ChatMessage>
        {
            ChatMessage.CreateSystemMessage(EnrichmentPrompt),
            ChatMessage.CreateUserMessage(text)
        };

        var options = new ChatCompletionOptions
        {
            MaxOutputTokenCount = 300,
            Temperature = 0.1f
        };

        // Retry with exponential backoff for rate limit errors
        for (int attempt = 0; attempt <= MaxRetries; attempt++)
        {
            try
            {
                var response = await _chat.CompleteChatAsync(messages, options, ct);
                var content  = response.Value.Content[0].Text ?? "";
                return ParseEnrichmentResponse(content);
            }
            catch (ClientResultException ex) when (ex.Status == 429 && attempt < MaxRetries)
            {
                _logger.LogWarning("Rate limited (attempt {Attempt}/{Max}), retrying in 60s...", 
                    attempt + 1, MaxRetries + 1);
                await Task.Delay(RetryDelay, ct);
            }
        }

        // If all retries failed, throw
        throw new InvalidOperationException("All retry attempts exhausted due to rate limiting");
    }

    private static (string? Summary, List<string> Entities, bool IsKey) ParseEnrichmentResponse(
        string response)
    {
        string? summary = null;
        var entities = new List<string>();
        bool isKey = false;

        // Parse structured response
        var lines = response.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        foreach (var line in lines)
        {
            if (line.StartsWith("SUMMARY:", StringComparison.OrdinalIgnoreCase))
                summary = line[8..].Trim();
            else if (line.StartsWith("ENTITIES:", StringComparison.OrdinalIgnoreCase))
                entities = line[9..].Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
            else if (line.StartsWith("KEY_PASSAGE:", StringComparison.OrdinalIgnoreCase))
                isKey = line.Contains("true", StringComparison.OrdinalIgnoreCase);
        }

        return (summary, entities, isKey);
    }

    private const string EnrichmentPrompt = """
        Analyze this FOMC document chunk and extract:
        1. A 1-sentence summary capturing the key economic/policy point
        2. Key entities: rates (e.g., "5.25%"), policy actions, economic indicators
        3. Whether this is a KEY_PASSAGE containing policy decisions

        Respond in EXACTLY this format:
        SUMMARY: <one sentence summary>
        ENTITIES: <comma-separated list>
        KEY_PASSAGE: true/false

        Focus on: interest rates, inflation figures, employment data, policy decisions, votes.
        """;
}
