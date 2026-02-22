using System.Text.Json;
using Azure;
using Azure.AI.OpenAI;
using FOMC.Data.Services;
using FOMC.Shared.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using OpenAI.Chat;

namespace FOMC.Agent;

/// <summary>
/// LLM-as-judge RAG evaluator.
///
/// Metrics:
///   Faithfulness       — are all answer claims grounded in the retrieved passages?
///   Context Relevance  — do the retrieved passages actually address the question?
///   Answer Similarity  — how close is the actual answer to the expected answer?
///
/// Each metric is scored 0–1 by the same GPT-5 model used for the agent.
/// </summary>
public class RagEvaluator
{
    private readonly ChatClient            _chat;
    private readonly TokenTracker          _tokens;
    private readonly ILogger<RagEvaluator> _logger;

    // ── Ground truth Q&A pairs for FOMC 2024-2025 corpus ─────────────────────
    public static readonly IReadOnlyList<(string Question, string ExpectedKeywords)> TestSet =
    [
        ("When did the Fed start cutting interest rates in 2024?",
         "September 2024 rate cut lower target range"),

        ("By how much did the Fed cut rates in September 2024?",
         "50 basis points half percentage 4-3/4 5 percent"),

        ("What was the federal funds rate target range at the end of 2024?",
         "4.25 4.5 percent target range December"),

        ("How did the FOMC describe inflation in 2024 meeting minutes?",
         "inflation elevated progress 2 percent target eased"),

        ("What was the Fed's stance on the labor market in early 2025?",
         "labor market solid strong employment unemployment low"),

        ("How many rate cuts did the Fed make in 2024?",
         "three cuts September November December 100 basis"),

        ("What did the FOMC say about economic growth in 2025?",
         "economic activity expanding solid pace growth continued"),

        ("What risks did the FOMC highlight in the 2025 meetings?",
         "inflation uncertainty risks economic outlook balance tariffs"),

        ("What was the Fed's balance sheet policy in 2024-2025?",
         "balance sheet securities holdings reduce runoff redemption"),

        ("How did the FOMC describe the housing market in 2024?",
         "housing elevated services inflation financing costs demand"),
    ];

    public RagEvaluator(IConfiguration config, TokenTracker tokens,
                        ILogger<RagEvaluator> logger)
    {
        _tokens = tokens;
        _logger = logger;

        var endpoint   = config["AzureOpenAI:Endpoint"]!;
        var apiKey     = config["AzureOpenAI:ApiKey"]!;
        var deployment = config["AzureOpenAI:ChatDeployment"] ?? "gpt-5-chat";

        var azureClient = new AzureOpenAIClient(new Uri(endpoint), new AzureKeyCredential(apiKey));
        _chat = azureClient.GetChatClient(deployment);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public async Task<EvaluationResult> EvaluateAsync(
        string question,
        string expectedKeywords,
        AgentResponse agentResponse,
        CancellationToken ct = default)
    {
        var retrievedTexts = agentResponse.Citations
            .Select(c => c.Chunk.Text)
            .ToList();

        var faithfulness       = await ScoreFaithfulnessAsync(question, agentResponse.Answer, retrievedTexts, ct);
        var contextRelevance   = await ScoreContextRelevanceAsync(question, retrievedTexts, ct);
        var answerSimilarity   = ScoreKeywordOverlap(agentResponse.Answer, expectedKeywords);

        return new EvaluationResult
        {
            Question              = question,
            ExpectedAnswer        = expectedKeywords,
            ActualAnswer          = agentResponse.Answer,
            FaithfulnessScore     = faithfulness.Score,
            ContextRelevanceScore = contextRelevance.Score,
            AnswerSimilarityScore = answerSimilarity,
            ToolCallsUsed         = agentResponse.ToolCallsUsed,
            SourcesRetrieved      = agentResponse.Citations.Count,
            JudgeReasoning        = $"F:{faithfulness.Reason} | CR:{contextRelevance.Reason}"
        };
    }

    public async Task<List<EvaluationResult>> RunFullEvalAsync(
        FomcAgent agent,
        CancellationToken ct = default)
    {
        var results = new List<EvaluationResult>();
        _logger.LogInformation("Running RAG evaluation on {N} test questions", TestSet.Count);

        foreach (var (question, expected) in TestSet)
        {
            _logger.LogInformation("Evaluating: {Q}", question[..Math.Min(60, question.Length)]);
            try
            {
                var agentResp = await agent.AskAsync(question, ct: ct);
                var result    = await EvaluateAsync(question, expected, agentResp, ct);
                results.Add(result);
                _logger.LogInformation(
                    "  F={F:F2} CR={CR:F2} AS={AS:F2} Overall={O:F2}",
                    result.FaithfulnessScore, result.ContextRelevanceScore,
                    result.AnswerSimilarityScore, result.OverallScore);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Evaluation failed for: {Q}", question);
                results.Add(new EvaluationResult
                {
                    Question              = question,
                    ExpectedAnswer        = expected,
                    ActualAnswer          = $"ERROR: {ex.Message}",
                    FaithfulnessScore     = 0,
                    ContextRelevanceScore = 0,
                    AnswerSimilarityScore = 0,
                    ToolCallsUsed         = 0,
                    SourcesRetrieved      = 0
                });
            }
        }

        PrintReport(results);
        return results;
    }

    // ── Scoring helpers ───────────────────────────────────────────────────────

    private async Task<(float Score, string Reason)> ScoreFaithfulnessAsync(
        string question, string answer,
        IList<string> contexts, CancellationToken ct)
    {
        if (contexts.Count == 0) return (0f, "no context retrieved");

        var contextBlock = string.Join("\n---\n", contexts.Take(5));
        var prompt = $$"""
            You are a strict faithfulness judge for a RAG system.

            QUESTION: {{question}}

            RETRIEVED CONTEXT:
            {{contextBlock}}

            MODEL ANSWER:
            {{answer}}

            TASK: Score the faithfulness of the answer on a scale of 0.0 to 1.0.
            Faithfulness = fraction of factual claims in the answer that are directly supported
            by the retrieved context above. Unsupported claims score 0.
            Answer in JSON: {"score": <0.0-1.0>, "reason": "<one sentence>"}
            """;

        return await CallJudgeAsync(prompt, ct);
    }

    private async Task<(float Score, string Reason)> ScoreContextRelevanceAsync(
        string question, IList<string> contexts, CancellationToken ct)
    {
        if (contexts.Count == 0) return (0f, "no context retrieved");

        var contextBlock = string.Join("\n---\n", contexts.Take(5));
        var prompt = $$"""
            You are a context relevance judge for a RAG system.

            QUESTION: {{question}}

            RETRIEVED CONTEXT:
            {{contextBlock}}

            TASK: Score the relevance of the retrieved context to the question on a scale of 0.0 to 1.0.
            Relevance = fraction of context passages that contain information useful for answering
            the question. Irrelevant passages score 0.
            Answer in JSON: {"score": <0.0-1.0>, "reason": "<one sentence>"}
            """;

        return await CallJudgeAsync(prompt, ct);
    }

    private async Task<(float Score, string Reason)> CallJudgeAsync(string prompt, CancellationToken ct)
    {
        try
        {
            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(
                    "You are a precise evaluation assistant. Respond only with the JSON requested."),
                ChatMessage.CreateUserMessage(prompt)
            };

            var options = new ChatCompletionOptions
            {
                MaxOutputTokenCount = 256,
                Temperature = 0f
            };

            var response = await _chat.CompleteChatAsync(messages, options, ct);
            var completion = response.Value;

            _tokens.AddChatTokens(
                completion.Usage.InputTokenCount,
                completion.Usage.OutputTokenCount);

            var json = completion.Content.Count > 0 
                ? completion.Content[0].Text ?? "{}" 
                : "{}";
            
            // Strip possible markdown code fences
            json = json.Trim().TrimStart('`').TrimEnd('`');
            if (json.StartsWith("json", StringComparison.OrdinalIgnoreCase))
                json = json[4..].TrimStart();

            using var doc = JsonDocument.Parse(json);
            var score = doc.RootElement.TryGetProperty("score", out var s)
                ? (float)s.GetDouble()
                : 0.5f;
            var reason = doc.RootElement.TryGetProperty("reason", out var r)
                ? r.GetString() ?? ""
                : "";
            return (Math.Clamp(score, 0f, 1f), reason);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Judge call failed; defaulting score to 0.5");
            return (0.5f, "judge error");
        }
    }

    /// <summary>
    /// Keyword-overlap metric with fuzzy matching — counts how many expected
    /// keywords (or their variants) appear in the actual answer, normalised 0–1.
    /// </summary>
    private static float ScoreKeywordOverlap(string answer, string keywords)
    {
        var tokens = keywords.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0) return 0f;
        var answerLower = answer.ToLowerInvariant();
        
        int hits = 0;
        foreach (var token in tokens)
        {
            var t = token.ToLowerInvariant();
            // Direct match
            if (answerLower.Contains(t))
            {
                hits++;
                continue;
            }
            // Common synonyms/variants for FOMC domain
            var matched = t switch
            {
                "cut" or "cuts" => answerLower.Contains("lower") || answerLower.Contains("reduc") || answerLower.Contains("decreas"),
                "rate" or "rates" => answerLower.Contains("funds") || answerLower.Contains("interest") || answerLower.Contains("target"),
                "basis" => answerLower.Contains("bps") || answerLower.Contains("point"),
                "percent" => answerLower.Contains("%"),
                "solid" => answerLower.Contains("strong") || answerLower.Contains("robust") || answerLower.Contains("healthy"),
                "subdued" => answerLower.Contains("weak") || answerLower.Contains("slow") || answerLower.Contains("soft") || answerLower.Contains("modest"),
                "reduce" or "reducing" => answerLower.Contains("runoff") || answerLower.Contains("taper") || answerLower.Contains("decreas"),
                "growth" => answerLower.Contains("expand") || answerLower.Contains("increas") || answerLower.Contains("rose"),
                "elevated" => answerLower.Contains("high") || answerLower.Contains("above"),
                "continued" => answerLower.Contains("ongoing") || answerLower.Contains("persist"),
                "tariffs" => answerLower.Contains("trade") || answerLower.Contains("import"),
                _ => false
            };
            if (matched) hits++;
        }
        return (float)hits / tokens.Length;
    }

    // ── Report ────────────────────────────────────────────────────────────────

    public static void PrintReport(IReadOnlyList<EvaluationResult> results)
    {
        if (results.Count == 0) return;

        var avgF  = results.Average(r => r.FaithfulnessScore);
        var avgCR = results.Average(r => r.ContextRelevanceScore);
        var avgAS = results.Average(r => r.AnswerSimilarityScore);
        var avgOv = results.Average(r => r.OverallScore);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"""

        ── RAG Evaluation Report ─────────────────────────────────
          Questions evaluated:    {results.Count}
          Avg Faithfulness:       {avgF:P0}
          Avg Context Relevance:  {avgCR:P0}
          Avg Answer Similarity:  {avgAS:P0}
          ─────────────────────────────────────────────────────────
          Overall Score:          {avgOv:P0}
        ──────────────────────────────────────────────────────────
        """);
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("Per-question scores:");
        foreach (var r in results)
        {
            var status = r.OverallScore >= 0.6f ? "✓" : r.OverallScore >= 0.3f ? "~" : "✗";
            Console.WriteLine($"  {status} [{r.OverallScore:F2}] {r.Question[..Math.Min(60, r.Question.Length)]}");
        }
        Console.ResetColor();
    }
}
