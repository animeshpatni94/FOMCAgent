using FOMC.Data.Factories;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Models;
using Microsoft.Extensions.Logging;
using OpenAI.Chat;
using System.Text.Json;

namespace FOMC.Data.Services;

/// <summary>
/// LLM-based reranker for improving retrieval precision (IReranker implementation).
/// Uses cross-encoder style scoring via LLM.
/// </summary>
public class Reranker : IReranker
{
    private readonly ChatClient _chat;
    private readonly ILogger<Reranker> _logger;

    public Reranker(
        IAzureOpenAIClientFactory factory,
        ILogger<Reranker> logger)
    {
        _chat = factory.CreateChatClient();
        _logger = logger;
    }

    /// <summary>
    /// Rerank search results based on relevance to the query.
    /// </summary>
    public async Task<List<SearchResult>> RerankAsync(
        string query,
        IReadOnlyList<SearchResult> candidates,
        int topN = 5,
        CancellationToken ct = default)
    {
        if (candidates.Count == 0) return [];
        if (candidates.Count <= topN) return candidates.ToList();

        var rerankPrompt = """
            You are a relevance scoring expert for Federal Reserve and FOMC documents.
            
            Given a query and a document, score the relevance from 0.0 to 1.0:
            - 1.0: Document directly answers the query with specific information
            - 0.8: Document is highly relevant and provides useful context
            - 0.5: Document is somewhat relevant, mentions related topics
            - 0.2: Document is tangentially related
            - 0.0: Document is not relevant to the query
            
            Respond with ONLY a JSON object: {"score": <float>, "reason": "<brief reason>"}
            """;

        var scoredResults = new List<(SearchResult Result, float Score)>();

        // Score each result
        foreach (var result in candidates)
        {
            try
            {
                var textSnippet = result.Chunk.Text[..Math.Min(1500, result.Chunk.Text.Length)];
                var userMessage = $"""
                    Query: {query}
                    
                    Document:
                    {textSnippet}
                    """;

                var messages = new List<ChatMessage>
                {
                    ChatMessage.CreateSystemMessage(rerankPrompt),
                    ChatMessage.CreateUserMessage(userMessage)
                };

                var options = new ChatCompletionOptions
                {
                    MaxOutputTokenCount = 100,
                    Temperature = 0f
                };

                var response = await _chat.CompleteChatAsync(messages, options, ct);
                var content = response.Value.Content[0].Text?.Trim();

                if (!string.IsNullOrEmpty(content))
                {
                    var scoreResult = JsonSerializer.Deserialize<RerankScore>(content);
                    if (scoreResult != null)
                    {
                        scoredResults.Add((result, scoreResult.Score));
                        continue;
                    }
                }

                // Fallback: use original score
                scoredResults.Add((result, result.Score));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Reranking failed for chunk {ChunkId}", result.Chunk.ChunkId);
                scoredResults.Add((result, result.Score));
            }
        }

        // Sort by rerank score and take top N
        var reranked = scoredResults
            .OrderByDescending(x => x.Score)
            .Take(topN)
            .Select(x => x.Result)
            .ToList();

        _logger.LogDebug("Reranked {Original} results to top {TopN}", candidates.Count, reranked.Count);
        return reranked;
    }

    private class RerankScore
    {
        public float Score { get; set; }
        public string? Reason { get; set; }
    }
}
