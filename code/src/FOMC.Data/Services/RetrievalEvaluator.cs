using FOMC.Data.Factories;
using FOMC.Shared.Abstractions;
using Microsoft.Extensions.Logging;
using OpenAI.Chat;
using System.Text.Json;

namespace FOMC.Data.Services;

/// <summary>
/// CRAG-style retrieval evaluator that assesses retrieval quality
/// and triggers corrective actions when needed (IRetrievalEvaluator implementation).
/// </summary>
public class RetrievalEvaluator : IRetrievalEvaluator
{
    private readonly ChatClient _chat;
    private readonly ILogger<RetrievalEvaluator> _logger;

    public RetrievalEvaluator(
        IAzureOpenAIClientFactory factory,
        ILogger<RetrievalEvaluator> logger)
    {
        _chat = factory.CreateChatClient();
        _logger = logger;
    }

    /// <summary>
    /// Evaluate retrieval quality using CRAG methodology.
    /// </summary>
    public async Task<RetrievalEvaluation> EvaluateAsync(
        string query,
        IReadOnlyList<EvaluationCandidate> candidates,
        CancellationToken ct = default)
    {
        if (candidates.Count == 0)
        {
            return new RetrievalEvaluation
            {
                Verdict = RetrievalVerdict.Incorrect,
                Confidence = 0f,
                Explanation = "No documents retrieved. Try broader search terms or different date range.",
                CandidateScores = [],
                SuggestedAction = "Retry with broader query or check available date range"
            };
        }

        var evaluationPrompt = """
            You are an expert evaluator of document retrieval quality for FOMC/Federal Reserve questions.
            
            Given a query and retrieved documents, assess if the documents can answer the query.
            
            For each document, score its relevance from 0-5:
            - 5: Directly answers the query with specific information
            - 4: Highly relevant, provides key context
            - 3: Moderately relevant, useful background
            - 2: Tangentially related
            - 1: Barely relevant
            - 0: Not relevant
            
            Then provide an overall verdict:
            - CORRECT: Documents fully support answering the query (avg score >= 3)
            - AMBIGUOUS: Documents partially relevant (avg score 1.5-3)
            - INCORRECT: Documents don't address the query (avg score < 1.5)
            
            Respond with ONLY a JSON object:
            {
                "verdict": "CORRECT" | "AMBIGUOUS" | "INCORRECT",
                "confidence": <0.0-1.0>,
                "explanation": "<brief explanation>",
                "scores": [{"id": "<chunk_id>", "score": <0-5>, "rationale": "<brief reason>"}],
                "suggested_action": "<action if not CORRECT>"
            }
            """;

        try
        {
            var docsContext = string.Join("\n---\n", candidates.Select(c =>
            {
                var text = c.Text[..Math.Min(600, c.Text.Length)];
                var summary = c.Summary != null ? $"\nSummary: {c.Summary}" : "";
                var keyFlag = c.IsKeyPassage ? " [KEY PASSAGE]" : "";
                return $"[ID: {c.ChunkId}]{keyFlag}{summary}\n{text}";
            }));

            var userMessage = $"""
                Query: {query}
                
                Retrieved Documents:
                {docsContext}
                """;

            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(evaluationPrompt),
                ChatMessage.CreateUserMessage(userMessage)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 600,
                Temperature = 0f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            var content = response.Value.Content[0].Text?.Trim();

            if (!string.IsNullOrEmpty(content))
            {
                var evalResult = JsonSerializer.Deserialize<EvaluationResponse>(content,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (evalResult != null)
                {
                    var verdict = evalResult.Verdict?.ToUpperInvariant() switch
                    {
                        "CORRECT" => RetrievalVerdict.Correct,
                        "AMBIGUOUS" => RetrievalVerdict.Ambiguous,
                        _ => RetrievalVerdict.Incorrect
                    };

                    var candidateScores = evalResult.Scores?
                        .Select(s => new CandidateRelevance(s.Id, s.Score, s.Rationale ?? ""))
                        .ToList() ?? [];

                    return new RetrievalEvaluation
                    {
                        Verdict = verdict,
                        Confidence = Math.Clamp(evalResult.Confidence, 0f, 1f),
                        Explanation = evalResult.Explanation ?? "",
                        CandidateScores = candidateScores,
                        SuggestedAction = evalResult.SuggestedAction
                    };
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Retrieval evaluation failed");
        }

        // Default: proceed with caution
        return new RetrievalEvaluation
        {
            Verdict = RetrievalVerdict.Ambiguous,
            Confidence = 0.5f,
            Explanation = "Evaluation failed, proceeding with available results",
            CandidateScores = candidates.Select(c => 
                new CandidateRelevance(c.ChunkId, 2, "Default score")).ToList(),
            SuggestedAction = null
        };
    }

    /// <summary>
    /// Get a refined query if initial retrieval was ambiguous or incorrect.
    /// </summary>
    public async Task<string?> GetRefinedQueryAsync(
        string originalQuery,
        RetrievalEvaluation evaluation,
        CancellationToken ct = default)
    {
        if (evaluation.Verdict == RetrievalVerdict.Correct)
        {
            return null; // No refinement needed
        }

        var refinementPrompt = """
            You are an expert at query refinement for FOMC document search.
            
            The original query did not retrieve good results. Based on the evaluation,
            generate a refined query that might work better.
            
            Guidelines:
            - Use FOMC/Federal Reserve terminology
            - Be more specific about dates, rates, or policy topics
            - Consider alternative phrasings
            
            Return ONLY the refined query string, nothing else.
            """;

        try
        {
            var userMessage = $"""
                Original Query: {originalQuery}
                
                Evaluation: {evaluation.Verdict}
                Explanation: {evaluation.Explanation}
                Suggested Action: {evaluation.SuggestedAction ?? "N/A"}
                """;

            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(refinementPrompt),
                ChatMessage.CreateUserMessage(userMessage)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 100,
                Temperature = 0.3f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            var refined = response.Value.Content[0].Text?.Trim();

            if (!string.IsNullOrEmpty(refined) && refined != originalQuery)
            {
                _logger.LogDebug("Query refined: {Original} -> {Refined}", originalQuery, refined);
                return refined;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Query refinement failed");
        }

        return null;
    }

    #region Response DTOs

    private class EvaluationResponse
    {
        public string? Verdict { get; set; }
        public float Confidence { get; set; }
        public string? Explanation { get; set; }
        public List<ScoreEntry>? Scores { get; set; }
        public string? SuggestedAction { get; set; }
    }

    private class ScoreEntry
    {
        public string Id { get; set; } = "";
        public int Score { get; set; }
        public string? Rationale { get; set; }
    }

    #endregion
}
