using System.ComponentModel;
using System.Text.Json;
using FOMC.Data.Services;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Constants;
using FOMC.Shared.Models;
using FOMC.Shared.Utilities;
using ModelContextProtocol.Server;

namespace FOMC.McpServer.Tools;

/// <summary>
/// MCP tool implementations exposed to the agent.
///
/// Tools provided:
///  1. search_fomc_documents      — semantic vector search (primary retrieval)
///  2. search_with_rerank         — bi-encoder + LLM reranking for precision
///  3. search_with_hyde           — HyDE: hypothetical document embedding for better recall
///  4. search_with_expansion      — Query expansion with RRF fusion
///  5. search_key_decisions       — Filter to policy decision passages only
///  6. search_decomposed          — Multi-hop query decomposition
///  7. search_with_context        — Step-back prompting for narrow queries
///  8. list_fomc_meetings         — enumerate available meetings
///  9. get_chunk_context          — expand a chunk ID to adjacent chunks
/// 10. evaluate_retrieval         — CRAG-style self-critique of search results
///
/// Uses interfaces for testability (DIP compliance).
/// </summary>
[McpServerToolType]
public class FomcSearchTools(
    IEmbeddingService embedder,
    IVectorStore vectorStore,
    IQueryEnhancer queryEnhancer,
    IReranker reranker,
    IRetrievalEvaluator? retrievalEvaluator = null)
{
    private static readonly JsonSerializerOptions JsonOpts =
        new() { WriteIndented = false };

    // NOTE: Key passage boosting is now handled by KeyPassageBoostingVectorStore decorator.
    // This applies a soft score boost (0.05) to IsKeyPassage=true chunks across ALL search tools,
    // reducing bias while prioritizing policy decisions. See FOMC.Data/Services/KeyPassageBoostingVectorStore.cs

    // ══════════════════════════════════════════════════════════════════════════
    // INPUT SANITIZATION (Prompt Injection Protection)
    // ══════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Sanitize query input to prevent prompt injection attacks.
    /// </summary>
    private static string SanitizeQuery(string query)
    {
        if (string.IsNullOrEmpty(query))
            return query;

        // Check for injection attempts and log if detected
        if (StringHelpers.ContainsInjectionAttempt(query))
        {
            // Sanitize but continue - the agent prompt is hardened
            return StringHelpers.SanitizeForPrompt(query);
        }

        return StringHelpers.SanitizeForPrompt(query);
    }

    // ── Tool 1: Semantic search ───────────────────────────────────────────────

    [McpServerTool(Name = "search_fomc_documents")]
    [Description(
        "Semantically search the FOMC document corpus (press statements and meeting minutes). " +
        "Returns the top-k most relevant text chunks with full citation metadata. " +
        "Use this before every answer to ground your response in source documents.")]
    public async Task<string> SearchFomcDocuments(
        [Description("Natural language search query, e.g. 'inflation expectations 2024'")]
        string query,

        [Description("Number of results to return (1–10). Default 8.")]
        int topK = 8,

        [Description("Optional filter: 'press_statement' or 'minutes'. Omit to search both.")]
        string? docType = null,

        [Description("Optional start date filter (ISO format, e.g. '2024-01-01').")]
        string? dateFrom = null,

        [Description("Optional end date filter (ISO format, e.g. '2025-12-31').")]
        string? dateTo = null)
    {
        // Sanitize input for security
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 10);

        var queryVector = await embedder.GetEmbeddingAsync(query);
        var results = await vectorStore.SearchAsync(
            queryVector, topK, docType, dateFrom, dateTo, hnswEf: 64);

        // NOTE: Key passage boosting is applied automatically by the IVectorStore decorator

        // ── Uncertainty Handling ─────────────────────────────────────────────
        // Signal to the agent when results may be unreliable
        var confidenceAssessment = AssessConfidence(results, query);

        if (results.Count == 0)
        {
            return JsonSerializer.Serialize(new
            {
                found       = 0,
                confidence  = "none",
                message     = "No relevant passages found for this query.",
                suggestion  = "Try broader search terms, check date range with list_fomc_meetings, or use search_with_hyde for conceptual queries.",
                results     = Array.Empty<object>()
            }, JsonOpts);
        }

        var payload = new
        {
            found      = results.Count,
            confidence = confidenceAssessment.Level,
            confidence_note = confidenceAssessment.Note,
            results = results.Select(r => new
            {
                citation       = r.Citation,
                score          = Math.Round(r.Score, 4),
                confidence     = r.Score > 0.85 ? "high" : r.Score > 0.70 ? "medium" : "low",
                doc_type       = r.Chunk.DocType,
                meeting_date   = r.Chunk.MeetingDate,
                section        = r.Chunk.SectionTitle,
                chunk_id       = r.Chunk.ChunkId,
                text           = r.Chunk.Text,
                source_url     = r.Chunk.SourceUrl,
                // Creative enrichment fields — help the agent prioritize and summarize
                summary        = r.Chunk.Summary,
                is_key_passage = r.Chunk.IsKeyPassage,
                key_entities   = r.Chunk.KeyEntities
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 2: Hybrid Search (Dense + Sparse/BM25) ───────────────────────────

    [McpServerTool(Name = "search_hybrid")]
    [Description(
        "HYBRID SEARCH: Combines semantic vector search with sparse BM25 keyword matching using Qdrant's native fusion. " +
        "Best for queries containing specific terms (rates, dates, technical terms like '5.25%', 'FOMC', 'quantitative tightening'). " +
        "Uses Reciprocal Rank Fusion (RRF) directly in Qdrant for optimal performance. " +
        "Based on Anthropic's Contextual Retrieval research showing 49% fewer retrieval failures for financial documents.")]
    public async Task<string> SearchHybrid(
        [Description("Natural language query, e.g. 'federal funds rate 5.25% March 2024'")]
        string query,

        [Description("Number of results to return (1–10). Default 8.")]
        int topK = 8,

        [Description("Weight for semantic search (0.0-1.0). Default 0.7 = 70% semantic, 30% keyword.")]
        float denseWeight = 0.7f,

        [Description("Optional filter: 'press_statement' or 'minutes'. Omit to search both.")]
        string? docType = null,

        [Description("Optional start date filter (ISO format, e.g. '2024-01-01').")]
        string? dateFrom = null,

        [Description("Optional end date filter (ISO format, e.g. '2025-12-31').")]
        string? dateTo = null)
    {
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 10);
        denseWeight = Math.Clamp(denseWeight, 0f, 1f);

        var queryVector = await embedder.GetEmbeddingAsync(query);
        var results = await vectorStore.HybridSearchAsync(
            queryVector, query, topK, denseWeight, docType, dateFrom, dateTo);

        var confidenceAssessment = AssessConfidence(results, query);

        if (results.Count == 0)
        {
            return JsonSerializer.Serialize(new
            {
                found       = 0,
                confidence  = "none",
                strategy    = "hybrid (dense + BM25 keyword)",
                message     = "No relevant passages found.",
                suggestion  = "Try search_with_hyde for conceptual queries or search_with_expansion for alternative terms.",
                results     = Array.Empty<object>()
            }, JsonOpts);
        }

        var payload = new
        {
            found      = results.Count,
            strategy   = $"hybrid (dense={denseWeight:P0}, keyword={1-denseWeight:P0})",
            confidence = confidenceAssessment.Level,
            confidence_note = confidenceAssessment.Note,
            note = "Results fused using Reciprocal Rank Fusion (RRF) for best of both semantic and keyword matching.",
            results = results.Select(r => new
            {
                citation       = r.Citation,
                score          = Math.Round(r.Score, 4),
                doc_type       = r.Chunk.DocType,
                meeting_date   = r.Chunk.MeetingDate,
                section        = r.Chunk.SectionTitle,
                chunk_id       = r.Chunk.ChunkId,
                text           = r.Chunk.Text,
                source_url     = r.Chunk.SourceUrl,
                summary        = r.Chunk.Summary,
                is_key_passage = r.Chunk.IsKeyPassage,
                key_entities   = r.Chunk.KeyEntities
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 3: Search with Reranking ────────────────────────────────────────

    [McpServerTool(Name = "search_with_rerank")]
    [Description(
        "PRECISION MODE: Retrieves more candidates then reranks them using LLM-based cross-encoder scoring. " +
        "Use this when you need the MOST relevant results, not just semantically similar ones. " +
        "Slower but more accurate than standard search. Best for specific factual questions.")]
    public async Task<string> SearchWithRerank(
        [Description("Natural language query")]
        string query,

        [Description("Number of final results after reranking (1–8). Default 5.")]
        int topK = 5,

        [Description("Optional document type filter: 'press_statement' or 'minutes'.")]
        string? docType = null)
    {
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 8);

        // Retrieve more candidates for reranking
        var candidateCount = topK * 3;
        var queryVector = await embedder.GetEmbeddingAsync(query);
        var candidates = await vectorStore.SearchAsync(queryVector, candidateCount, docType, hnswEf: 64);

        if (candidates.Count == 0)
        {
            return JsonSerializer.Serialize(new
            {
                found = 0,
                message = "No candidates found for reranking.",
                results = Array.Empty<object>()
            }, JsonOpts);
        }

        // Rerank using LLM-based cross-encoder scoring
        var reranked = await reranker.RerankAsync(query, candidates, topK);

        var payload = new
        {
            found = reranked.Count,
            strategy = "bi-encoder + LLM rerank",
            note = $"Retrieved {candidates.Count} candidates, reranked to top {topK}",
            results = reranked.Select((r, i) => new
            {
                rerank_position = i + 1,
                citation = r.Citation,
                original_score = Math.Round(r.Score, 4),
                meeting_date = r.Chunk.MeetingDate,
                section = r.Chunk.SectionTitle,
                summary = r.Chunk.Summary,
                text = r.Chunk.Text,
                is_key_passage = r.Chunk.IsKeyPassage,
                source_url = r.Chunk.SourceUrl
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 3: List available meetings ──────────────────────────────────────

    [McpServerTool(Name = "list_fomc_meetings")]
    [Description(
        "Returns a list of all FOMC meeting dates available in the corpus. " +
        "Use this to understand what time period is covered before making date-filtered queries.")]
    public async Task<string> ListFomcMeetings()
    {
        var dates = await vectorStore.GetMeetingDatesAsync();
        var total = await vectorStore.CountAsync();

        return JsonSerializer.Serialize(new
        {
            total_chunks   = total,
            meeting_count  = dates.Count,
            meeting_dates  = dates,
            date_range     = dates.Count > 0
                ? new { from = dates[0], to = dates[^1] }
                : null
        }, JsonOpts);
    }

    // ── Tool 3: Expand chunk context ─────────────────────────────────────────

    [McpServerTool(Name = "get_chunk_context")]
    [Description(
        "Retrieves a specific chunk by its ID, useful for expanding context around a retrieved passage. " +
        "Use this when a search result is truncated or you need the surrounding text for completeness.")]
    public async Task<string> GetChunkContext(
        [Description("The chunk_id UUID returned by search_fomc_documents.")]
        string chunkId)
    {
        var chunk = await vectorStore.GetChunkAsync(chunkId);

        if (chunk is null)
            return JsonSerializer.Serialize(new
            {
                found   = false,
                message = $"Chunk '{chunkId}' not found."
            }, JsonOpts);

        return JsonSerializer.Serialize(new
        {
            found          = true,
            citation       = new SearchResult { Chunk = chunk, Score = 1f }.Citation,
            chunk_id       = chunk.ChunkId,
            doc_type       = chunk.DocType,
            meeting_date   = chunk.MeetingDate,
            chunk_index    = chunk.ChunkIndex,
            total_chunks   = chunk.TotalChunks,
            section        = chunk.SectionTitle,
            text           = chunk.Text,
            source_url     = chunk.SourceUrl,
            // Enrichment fields
            summary        = chunk.Summary,
            is_key_passage = chunk.IsKeyPassage,
            key_entities   = chunk.KeyEntities
        }, JsonOpts);
    }

    // ── Tool 4: Search key policy passages only ──────────────────────────────

    [McpServerTool(Name = "search_key_decisions")]
    [Description(
        "Search only chunks flagged as KEY PASSAGES containing policy decisions, votes, or rate changes. " +
        "Use this when the user asks specifically about Fed decisions, rate changes, or votes. " +
        "Returns fewer but more relevant results than general search.")]
    public async Task<string> SearchKeyDecisions(
        [Description("Natural language search query about policy decisions.")]
        string query,

        [Description("Number of results to return (1–8). Default 5.")]
        int topK = 5,

        [Description("Optional meeting date filter (ISO format).")]
        string? meetingDate = null)
    {
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 8);

        var queryVector = await embedder.GetEmbeddingAsync(query);
        
        // Search more results, then filter to key passages
        var allResults = await vectorStore.SearchAsync(
            queryVector, topK * 3, dateFrom: meetingDate, dateTo: meetingDate, hnswEf: 64);

        var keyResults = allResults
            .Where(r => r.Chunk.IsKeyPassage)
            .Take(topK)
            .ToList();

        if (keyResults.Count == 0)
        {
            return JsonSerializer.Serialize(new
            {
                found   = 0,
                message = "No key policy passages found for this query. Try search_fomc_documents for broader results.",
                results = Array.Empty<object>()
            }, JsonOpts);
        }

        var payload = new
        {
            found   = keyResults.Count,
            note    = "These are KEY PASSAGES containing policy decisions or votes.",
            results = keyResults.Select(r => new
            {
                citation       = r.Citation,
                score          = Math.Round(r.Score, 4),
                meeting_date   = r.Chunk.MeetingDate,
                summary        = r.Chunk.Summary,
                text           = r.Chunk.Text,
                key_entities   = r.Chunk.KeyEntities,
                source_url     = r.Chunk.SourceUrl
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // ADVANCED RAG RETRIEVAL TECHNIQUES
    // ══════════════════════════════════════════════════════════════════════════

    // ── Tool 5: HyDE Search (Hypothetical Document Embedding) ────────────────

    [McpServerTool(Name = "search_with_hyde")]
    [Description(
        "ADVANCED: Uses HyDE (Hypothetical Document Embedding) for better recall on conceptual queries. " +
        "Generates a hypothetical FOMC document excerpt, then finds real documents similar to it. " +
        "Use this when standard search returns poor results, or for high-level questions like " +
        "'What was the Fed's overall stance on X?'")]
    public async Task<string> SearchWithHyde(
        [Description("Natural language query about FOMC policy")]
        string query,

        [Description("Number of results (1–10). Default 8.")]
        int topK = 8)
    {
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 8);

        var hyde = await queryEnhancer.GenerateHydeEmbeddingAsync(query);
        var results = await vectorStore.SearchAsync(hyde.Embedding, topK, hnswEf: 64);

        var payload = new
        {
            found = results.Count,
            strategy = hyde.Strategy,
            hypothetical_excerpt = hyde.HypotheticalDocument != null 
                ? StringHelpers.Truncate(hyde.HypotheticalDocument, 200) 
                : null,
            note = hyde.Strategy == "hyde" 
                ? "Used HyDE: searched for docs similar to a generated hypothetical excerpt"
                : "HyDE not applied, used direct embedding",
            results = results.Select(r => new
            {
                citation       = r.Citation,
                score          = Math.Round(r.Score, 4),
                meeting_date   = r.Chunk.MeetingDate,
                section        = r.Chunk.SectionTitle,
                summary        = r.Chunk.Summary,
                text           = r.Chunk.Text,
                source_url     = r.Chunk.SourceUrl
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 6: Query Expansion Search with RRF ──────────────────────────────

    [McpServerTool(Name = "search_with_expansion")]
    [Description(
        "ADVANCED: Expands the query into multiple phrasings and merges results using Reciprocal Rank Fusion. " +
        "Use this when you suspect the user's terminology differs from FOMC language. " +
        "Example: 'rate hike' expands to include 'interest rate increase', 'tightening monetary policy'.")]
    public async Task<string> SearchWithExpansion(
        [Description("Natural language query")]
        string query,

        [Description("Number of final results after fusion (1–10). Default 8.")]
        int topK = 8)
    {
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 10);

        var expansion = await queryEnhancer.ExpandQueryAsync(query);
        var allResultLists = new List<IList<SearchResult>>();

        // Search each expanded query
        foreach (var expandedQuery in expansion.ExpandedQueries)
        {
            var vec = await embedder.GetEmbeddingAsync(expandedQuery);
            var results = await vectorStore.SearchAsync(vec, topK, hnswEf: 64);
            allResultLists.Add(results.ToList());
        }

        // Reciprocal Rank Fusion
        var fused = QueryEnhancer.ReciprocalRankFusion(
            allResultLists,
            r => r.Chunk.ChunkId,
            k: 60,
            topK: topK);

        var payload = new
        {
            found = fused.Count,
            strategy = expansion.Strategy,
            queries_used = expansion.ExpandedQueries,
            note = $"Searched {expansion.ExpandedQueries.Count} query variants, merged with RRF",
            results = fused.Select(r => new
            {
                citation       = r.Citation,
                score          = Math.Round(r.Score, 4),
                meeting_date   = r.Chunk.MeetingDate,
                section        = r.Chunk.SectionTitle,
                summary        = r.Chunk.Summary,
                text           = r.Chunk.Text,
                source_url     = r.Chunk.SourceUrl
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 7: Multi-hop Decomposed Search ──────────────────────────────────

    [McpServerTool(Name = "search_decomposed")]
    [Description(
        "ADVANCED: For complex multi-part questions, decomposes into sub-questions and searches each. " +
        "Use for comparisons ('How did X change from 2024 to 2025?') or multi-aspect queries. " +
        "Returns results grouped by sub-question.")]
    public async Task<string> SearchDecomposed(
        [Description("Complex query to decompose")]
        string query,

        [Description("Results per sub-question (1–8). Default 5.")]
        int topKPerSubQuery = 5)
    {
        query = SanitizeQuery(query);
        topKPerSubQuery = Math.Clamp(topKPerSubQuery, 1, 8);

        var decomposition = await queryEnhancer.DecomposeQueryAsync(query);
        var groupedResults = new List<object>();

        foreach (var subQuery in decomposition.SubQueries)
        {
            var vec = await embedder.GetEmbeddingAsync(subQuery);
            var results = await vectorStore.SearchAsync(vec, topKPerSubQuery, hnswEf: 64);

            groupedResults.Add(new
            {
                sub_query = subQuery,
                results = results.Select(r => new
                {
                    citation     = r.Citation,
                    score        = Math.Round(r.Score, 4),
                    meeting_date = r.Chunk.MeetingDate,
                    summary      = r.Chunk.Summary,
                    text         = r.Chunk.Text
                })
            });
        }

        var payload = new
        {
            original_query = query,
            strategy = decomposition.Strategy,
            sub_queries_count = decomposition.SubQueries.Count,
            note = decomposition.Strategy == "decomposed"
                ? "Query was decomposed into sub-questions for comprehensive retrieval"
                : "Query was simple enough, no decomposition needed",
            grouped_results = groupedResults
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 8: Step-Back Search ─────────────────────────────────────────────

    [McpServerTool(Name = "search_with_context")]
    [Description(
        "ADVANCED: For specific/narrow queries, first searches a broader 'step-back' question for context, " +
        "then searches the specific query. Returns both contextual and specific results. " +
        "Use when specific queries return too few results.")]
    public async Task<string> SearchWithContext(
        [Description("Specific query that may need broader context")]
        string query,

        [Description("Results for context query (1–3). Default 2.")]
        int contextTopK = 2,

        [Description("Results for specific query (1–5). Default 3.")]
        int specificTopK = 3)
    {
        query = SanitizeQuery(query);
        contextTopK = Math.Clamp(contextTopK, 1, 3);
        specificTopK = Math.Clamp(specificTopK, 1, 5);

        var stepBack = await queryEnhancer.GenerateStepBackQueryAsync(query);

        // Always search the specific query
        var specificVec = await embedder.GetEmbeddingAsync(query);
        var specificResults = await vectorStore.SearchAsync(specificVec, specificTopK, hnswEf: 64);

        object? contextSection = null;
        if (stepBack.StepBackQuery != null)
        {
            var contextVec = await embedder.GetEmbeddingAsync(stepBack.StepBackQuery);
            var contextResults = await vectorStore.SearchAsync(contextVec, contextTopK, hnswEf: 64);

            contextSection = new
            {
                step_back_query = stepBack.StepBackQuery,
                results = contextResults.Select(r => new
                {
                    citation     = r.Citation,
                    meeting_date = r.Chunk.MeetingDate,
                    summary      = r.Chunk.Summary,
                    text         = r.Chunk.Text
                })
            };
        }

        var payload = new
        {
            strategy = stepBack.Strategy,
            note = stepBack.Strategy == "step-back"
                ? "Included broader context to help answer the specific question"
                : "Query was general enough, no step-back needed",
            context = contextSection,
            specific = new
            {
                query = query,
                results = specificResults.Select(r => new
                {
                    citation       = r.Citation,
                    score          = Math.Round(r.Score, 4),
                    meeting_date   = r.Chunk.MeetingDate,
                    summary        = r.Chunk.Summary,
                    text           = r.Chunk.Text,
                    source_url     = r.Chunk.SourceUrl
                })
            }
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CONFIDENCE ASSESSMENT (Uncertainty Handling)
    // ══════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Assess overall confidence in search results to help agent handle uncertainty.
    /// Uses thresholds from Defaults constants for consistency.
    /// </summary>
    private static (string Level, string Note) AssessConfidence(
        IReadOnlyList<SearchResult> results,
        string query)
    {
        if (results.Count == 0)
            return ("none", "No results found. The corpus may not contain information on this topic.");

        var topScore = results[0].Score;
        var avgScore = results.Average(r => r.Score);
        var hasKeyPassage = results.Any(r => r.Chunk.IsKeyPassage);
        var scoreSpread = results.Count > 1 ? results[0].Score - results[^1].Score : 0;

        // High confidence: strong top match + good spread (clear winner)
        if (topScore > Defaults.HighConfidenceThreshold && scoreSpread > 0.1)
            return ("high", "Strong match found with clear relevance differentiation.");

        // High confidence: key passage with decent score
        if (topScore > 0.80 && hasKeyPassage)
            return ("high", "Matched a key policy passage with strong relevance.");

        // Medium confidence: decent scores but less differentiation
        if (topScore > Defaults.MediumConfidenceThreshold && avgScore > 0.70)
            return ("medium", "Reasonable matches found. Consider verifying with additional searches.");

        // Low confidence: weak matches or no differentiation
        if (topScore > Defaults.LowConfidenceThreshold)
            return ("low", 
                "Matches are weak. Results may be tangentially related. " +
                "Consider rephrasing the query or using search_with_hyde for better recall.");

        // Very low confidence
        return ("very_low", 
            "Results have poor relevance scores. The corpus likely doesn't contain " +
            "direct information on this topic. Recommend stating this limitation to the user.");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CRAG-STYLE SELF-CRITIQUE (Corrective RAG)
    // ══════════════════════════════════════════════════════════════════════════

    // ── Tool 10: Evaluate Retrieval Quality ──────────────────────────────────

    [McpServerTool(Name = "evaluate_retrieval")]
    [Description(
        "SELF-CRITIQUE: Evaluate whether retrieved documents are truly relevant to the query. " +
        "Use this when you're uncertain about result quality or before generating a final answer. " +
        "Returns CORRECT (proceed), AMBIGUOUS (refine query), or INCORRECT (try different search). " +
        "Implements CRAG (Corrective RAG) pattern for quality assurance.")]
    public async Task<string> EvaluateRetrieval(
        [Description("The original query that was searched")]
        string query,

        [Description("Comma-separated chunk IDs to evaluate (from search results)")]
        string chunkIds)
    {
        query = SanitizeQuery(query);
        
        if (retrievalEvaluator is null)
        {
            return JsonSerializer.Serialize(new
            {
                error = "Retrieval evaluator not configured",
                fallback_recommendation = "Use confidence scores from search results instead"
            }, JsonOpts);
        }

        var ids = chunkIds.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var candidates = new List<EvaluationCandidate>();

        foreach (var id in ids.Take(5)) // Limit to 5 for cost efficiency
        {
            var chunk = await vectorStore.GetChunkAsync(id.Trim());
            if (chunk is not null)
            {
                candidates.Add(new EvaluationCandidate(
                    chunk.ChunkId,
                    chunk.Text,
                    chunk.Summary,
                    0.8f, // Default score for manual evaluation
                    chunk.IsKeyPassage));
            }
        }

        if (candidates.Count == 0)
        {
            return JsonSerializer.Serialize(new
            {
                verdict = "INCORRECT",
                explanation = "None of the specified chunk IDs were found",
                suggestion = "Re-run search with different parameters"
            }, JsonOpts);
        }

        var evaluation = await retrievalEvaluator.EvaluateAsync(query, candidates);

        // Get refined query if needed
        string? refinedQuery = null;
        if (evaluation.Verdict != RetrievalVerdict.Correct)
        {
            refinedQuery = await retrievalEvaluator.GetRefinedQueryAsync(query, evaluation);
        }

        var payload = new
        {
            verdict = evaluation.Verdict.ToString().ToUpperInvariant(),
            confidence = Math.Round(evaluation.Confidence, 3),
            explanation = evaluation.Explanation,
            action_guidance = evaluation.Verdict switch
            {
                RetrievalVerdict.Correct => "Proceed with answer generation using these results.",
                RetrievalVerdict.Ambiguous => "Results are partial. Consider refining the query or searching with additional tools.",
                RetrievalVerdict.Incorrect => "Results are not relevant. Try a different search strategy.",
                _ => "Unknown verdict"
            },
            suggested_action = evaluation.SuggestedAction,
            refined_query = refinedQuery,
            candidate_scores = evaluation.CandidateScores.Select(c => new
            {
                chunk_id = c.ChunkId,
                relevance_score = c.RelevanceScore,
                rationale = c.Rationale
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    // ── Tool 11: Search with Self-Critique ───────────────────────────────────

    [McpServerTool(Name = "search_with_critique")]
    [Description(
        "SELF-REFLECTIVE RAG: Searches the corpus AND automatically evaluates result quality. " +
        "If results are inadequate, automatically tries alternative search strategies. " +
        "Returns both results and quality assessment. Best for important queries.")]
    public async Task<string> SearchWithCritique(
        [Description("Natural language query")]
        string query,

        [Description("Number of results (1–8). Default 5.")]
        int topK = 5)
    {
        query = SanitizeQuery(query);
        topK = Math.Clamp(topK, 1, 8);

        // Initial search
        var queryVector = await embedder.GetEmbeddingAsync(query);
        var results = await vectorStore.SearchAsync(queryVector, topK, hnswEf: 64);

        // Quick heuristic evaluation
        var topScore = results.Count > 0 ? results[0].Score : 0;
        var avgScore = results.Count > 0 ? results.Average(r => r.Score) : 0;
        var hasKeyPassage = results.Any(r => r.Chunk.IsKeyPassage);

        string verdict;
        string strategy = "semantic_search";
        IReadOnlyList<SearchResult> finalResults = results;

        // Self-critique: determine if results are adequate
        if (topScore > 0.85 || (topScore > 0.75 && hasKeyPassage))
        {
            verdict = "CORRECT";
        }
        else if (topScore > 0.65)
        {
            verdict = "AMBIGUOUS";
            // Try HyDE as fallback
            var hyde = await queryEnhancer.GenerateHydeEmbeddingAsync(query);
            var hydeResults = await vectorStore.SearchAsync(hyde.Embedding, topK, hnswEf: 64);
            
            if (hydeResults.Count > 0 && hydeResults[0].Score > topScore)
            {
                finalResults = hydeResults;
                strategy = "hyde_fallback";
                verdict = hydeResults[0].Score > 0.80 ? "CORRECT" : "AMBIGUOUS";
            }
        }
        else
        {
            verdict = "INCORRECT";
            // Try query expansion as last resort
            var expansion = await queryEnhancer.ExpandQueryAsync(query);
            var allResultLists = new List<IList<SearchResult>>();

            foreach (var expandedQuery in expansion.ExpandedQueries.Take(3))
            {
                var vec = await embedder.GetEmbeddingAsync(expandedQuery);
                var res = await vectorStore.SearchAsync(vec, topK, hnswEf: 64);
                allResultLists.Add(res.ToList());
            }

            var fused = QueryEnhancer.ReciprocalRankFusion(
                allResultLists, r => r.Chunk.ChunkId, k: 60, topK: topK);

            if (fused.Count > 0 && fused[0].Score > topScore)
            {
                finalResults = fused;
                strategy = "expansion_fallback";
                verdict = fused[0].Score > 0.70 ? "AMBIGUOUS" : "INCORRECT";
            }
        }

        var payload = new
        {
            verdict,
            strategy_used = strategy,
            self_critique = new
            {
                initial_top_score = Math.Round(topScore, 4),
                final_top_score = finalResults.Count > 0 ? Math.Round(finalResults[0].Score, 4) : 0,
                fallback_applied = strategy != "semantic_search",
                recommendation = verdict switch
                {
                    "CORRECT" => "Results are reliable. Proceed with answer generation.",
                    "AMBIGUOUS" => "Results may be incomplete. Consider mentioning uncertainty to user.",
                    "INCORRECT" => "Could not find relevant results. State this limitation clearly.",
                    _ => "Unknown"
                }
            },
            found = finalResults.Count,
            results = finalResults.Select(r => new
            {
                citation = r.Citation,
                score = Math.Round(r.Score, 4),
                meeting_date = r.Chunk.MeetingDate,
                summary = r.Chunk.Summary,
                text = r.Chunk.Text,
                is_key_passage = r.Chunk.IsKeyPassage,
                source_url = r.Chunk.SourceUrl
            })
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }
}
