using FOMC.Data.Factories;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Models;
using Microsoft.Extensions.Logging;
using OpenAI.Chat;
using System.Text.Json;

namespace FOMC.Data.Services;

/// <summary>
/// Query Enhancement Methods (IQueryEnhancer implementation):
/// - HyDE: Generates hypothetical documents to improve retrieval
/// - Query Decomposition: Breaks complex queries into sub-queries
/// - Query Expansion: Adds related terms and synonyms
/// - Step-Back: Generates broader conceptual queries
/// </summary>
public class QueryEnhancer : IQueryEnhancer
{
    private readonly ChatClient _chat;
    private readonly IEmbeddingService _embeddings;
    private readonly ILogger<QueryEnhancer> _logger;

    public QueryEnhancer(
        IAzureOpenAIClientFactory factory,
        IEmbeddingService embeddings,
        ILogger<QueryEnhancer> logger)
    {
        _chat = factory.CreateChatClient();
        _embeddings = embeddings;
        _logger = logger;
    }

    #region HyDE (Hypothetical Document Embeddings)

    /// <summary>
    /// Generate embedding for a hypothetical document that answers the query.
    /// </summary>
    public async Task<HydeResult> GenerateHydeEmbeddingAsync(
        string query,
        CancellationToken ct = default)
    {
        var hydePrompt = """
            You are an expert on Federal Reserve monetary policy and FOMC communications.
            Given the following question, write a detailed passage that would perfectly answer it.
            The passage should sound like it comes from an official FOMC document.
            Write only the passage, no introduction or conclusion.
            """;

        string? hypotheticalDoc = null;
        float[] embedding;

        try
        {
            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(hydePrompt),
                ChatMessage.CreateUserMessage(query)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 300,
                Temperature = 0.7f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            hypotheticalDoc = response.Value.Content[0].Text?.Trim();

            if (string.IsNullOrEmpty(hypotheticalDoc))
            {
                _logger.LogWarning("HyDE generation returned empty result, using original query");
                embedding = await _embeddings.GetEmbeddingAsync(query, ct);
            }
            else
            {
                _logger.LogDebug("HyDE generated document: {Doc}", 
                    hypotheticalDoc[..Math.Min(100, hypotheticalDoc.Length)]);
                var combined = $"{query}\n\n{hypotheticalDoc}";
                embedding = await _embeddings.GetEmbeddingAsync(combined, ct);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "HyDE generation failed, falling back to original query");
            embedding = await _embeddings.GetEmbeddingAsync(query, ct);
        }

        return new HydeResult
        {
            OriginalQuery = query,
            HypotheticalDocument = hypotheticalDoc,
            Embedding = embedding,
            Strategy = "hyde_v1"
        };
    }

    #endregion

    #region Query Decomposition

    /// <summary>
    /// Break complex queries into simpler sub-queries for multi-hop retrieval.
    /// </summary>
    public async Task<QueryDecomposition> DecomposeQueryAsync(
        string query,
        CancellationToken ct = default)
    {
        var decompositionPrompt = """
            You are an expert query analyzer for FOMC and Federal Reserve documents.
            Given a complex question, break it down into simpler sub-questions that:
            1. Can be answered independently
            2. Together provide information to answer the original question
            3. Are specific enough for semantic search
            
            Return ONLY a JSON array of strings with 2-4 sub-questions.
            Example: ["What was the inflation rate?", "What did the Fed say about employment?"]
            """;

        var subQueries = new List<string> { query };

        try
        {
            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(decompositionPrompt),
                ChatMessage.CreateUserMessage(query)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 200,
                Temperature = 0.3f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            var content = response.Value.Content[0].Text?.Trim();

            if (!string.IsNullOrEmpty(content))
            {
                var parsed = JsonSerializer.Deserialize<List<string>>(content);
                if (parsed != null && parsed.Count > 0)
                {
                    subQueries = parsed;
                    _logger.LogDebug("Query decomposed into {Count} sub-queries", subQueries.Count);
                }
            }
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Failed to parse decomposition result");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Query decomposition failed");
        }

        return new QueryDecomposition
        {
            OriginalQuery = query,
            SubQueries = subQueries,
            Strategy = "decomposition_v1"
        };
    }

    #endregion

    #region Query Expansion

    /// <summary>
    /// Expand query with synonyms, related terms, and financial concepts.
    /// </summary>
    public async Task<QueryExpansion> ExpandQueryAsync(
        string query,
        CancellationToken ct = default)
    {
        var expansionPrompt = """
            You are an expert on Federal Reserve terminology and monetary policy.
            Given a query, generate 2-3 alternative phrasings that:
            1. Use synonyms for key terms
            2. Include related financial concepts
            3. Use different but equivalent phrasing
            
            Return ONLY a JSON array of alternative query strings.
            Example: ["Fed interest rate decision", "FOMC policy rate change", "monetary policy tightening"]
            """;

        var expandedQueries = new List<string> { query };

        try
        {
            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(expansionPrompt),
                ChatMessage.CreateUserMessage(query)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 150,
                Temperature = 0.5f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            var content = response.Value.Content[0].Text?.Trim();

            if (!string.IsNullOrEmpty(content))
            {
                var parsed = JsonSerializer.Deserialize<List<string>>(content);
                if (parsed != null && parsed.Count > 0)
                {
                    expandedQueries.AddRange(parsed);
                    _logger.LogDebug("Query expanded to {Count} variations", expandedQueries.Count);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Query expansion failed");
        }

        return new QueryExpansion
        {
            OriginalQuery = query,
            ExpandedQueries = expandedQueries,
            Strategy = "expansion_v1"
        };
    }

    #endregion

    #region Step-Back Prompting

    /// <summary>
    /// Generate a higher-level "step-back" query for broader context retrieval.
    /// </summary>
    public async Task<StepBackResult> GenerateStepBackQueryAsync(
        string query,
        CancellationToken ct = default)
    {
        var stepBackPrompt = """
            You are an expert analyst of Federal Reserve policy.
            Given a specific question, generate a broader "step-back" question that:
            1. Asks about the general principle or concept behind the specific question
            2. Would retrieve helpful background context
            3. Is still relevant to FOMC and monetary policy
            
            Return ONLY the step-back question, nothing else.
            """;

        string? stepBackQuery = null;

        try
        {
            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(stepBackPrompt),
                ChatMessage.CreateUserMessage(query)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 100,
                Temperature = 0.5f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            stepBackQuery = response.Value.Content[0].Text?.Trim();
            
            _logger.LogDebug("Step-back query: {StepBack}", stepBackQuery);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Step-back generation failed");
        }

        return new StepBackResult
        {
            OriginalQuery = query,
            StepBackQuery = stepBackQuery,
            Strategy = "stepback_v1"
        };
    }

    #endregion

    #region Reciprocal Rank Fusion (static utility)

    /// <summary>
    /// Reciprocal Rank Fusion (RRF) for merging multiple result lists.
    /// Used when combining results from query expansion or multi-query retrieval.
    /// </summary>
    /// <param name="resultLists">Multiple ranked lists to fuse.</param>
    /// <param name="idSelector">Function to extract unique ID from each result.</param>
    /// <param name="k">RRF constant (default 60).</param>
    /// <param name="topK">Number of results to return.</param>
    public static List<T> ReciprocalRankFusion<T>(
        IEnumerable<IEnumerable<T>> resultLists,
        Func<T, string> idSelector,
        int k = 60,
        int topK = 5)
    {
        var scores = new Dictionary<string, (double Score, T Item)>();

        foreach (var list in resultLists)
        {
            int rank = 0;
            foreach (var item in list)
            {
                var id = idSelector(item);
                var rrfScore = 1.0 / (k + rank + 1);

                if (scores.TryGetValue(id, out var existing))
                {
                    scores[id] = (existing.Score + rrfScore, existing.Item);
                }
                else
                {
                    scores[id] = (rrfScore, item);
                }
                rank++;
            }
        }

        return scores.Values
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x => x.Item)
            .ToList();
    }

    #endregion
}
