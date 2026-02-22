using FOMC.Data.Services;
using FOMC.Shared.Models;
using Microsoft.Extensions.Logging;

namespace FOMC.Data.Ingestion;

/// <summary>
/// Orchestrates the full download → parse → chunk → enrich → embed → index pipeline.
///
/// CREATIVE ENHANCEMENTS:
///  1. Chunk enrichment: LLM generates summaries + extracts key entities at ingest time
///  2. Summary-augmented embeddings: embed (summary + text) for better semantic recall
///  3. Key passage detection: flags chunks containing policy decisions
///  4. Document summaries: shared context across all chunks in a document
///  5. Sparse vectors: BM25 stored directly in Qdrant for native hybrid search
///
/// Usage:
///   var pipeline = host.Services.GetRequiredService&lt;IngestionPipeline&gt;();
///   await pipeline.RunAsync();
///
/// Idempotent: re-running updates existing Qdrant points via upsert.
/// </summary>
public class IngestionPipeline
{
    private readonly FomcDownloader   _downloader;
    private readonly EmbeddingService _embedder;
    private readonly QdrantService    _qdrant;
    private readonly ChunkEnricher    _enricher;
    private readonly ILogger<IngestionPipeline> _logger;

    // Embed + upsert in batches to balance throughput vs. memory
    private const int EmbedBatchSize = 16;
    private const int EnrichBatchSize = 8;

    public IngestionPipeline(
        FomcDownloader   downloader,
        EmbeddingService embedder,
        QdrantService    qdrant,
        ChunkEnricher    enricher,
        ILogger<IngestionPipeline> logger)
    {
        _downloader = downloader;
        _embedder   = embedder;
        _qdrant     = qdrant;
        _enricher   = enricher;
        _logger     = logger;
    }

    public async Task RunAsync(
        IEnumerable<string>? datesToProcess = null,
        CancellationToken ct = default)
    {
        // Ensure collection exists with dense + sparse vector support
        _logger.LogInformation("Ensuring Qdrant collection exists with {Dims} dimensions + sparse vectors...", _embedder.Dimensions);
        await _qdrant.EnsureCollectionAsync(_embedder.Dimensions, ct);
        _logger.LogInformation("Qdrant collection ready (hybrid search enabled).");

        var dates = datesToProcess?.ToArray() ?? FomcDownloader.MeetingDates;
        _logger.LogInformation("Starting ingestion for {Count} meetings.", dates.Length);

        int totalChunks = 0;
        int meetingIndex = 0;

        foreach (var date in dates)
        {
            ct.ThrowIfCancellationRequested();
            meetingIndex++;
            _logger.LogInformation("[{Index}/{Total}] Downloading documents for meeting {Date}...", 
                meetingIndex, dates.Length, date);

            var docs = await _downloader.DownloadMeetingAsync(date, ct);
            _logger.LogInformation("[{Index}/{Total}] Downloaded {DocCount} documents for {Date}.", 
                meetingIndex, dates.Length, docs.Count, date);

            foreach (var (docType, _, url, html) in docs)
            {
                // ISO date e.g. "2024-01-31"
                var isoDate = $"{date[..4]}-{date[4..6]}-{date[6..]}";

                _logger.LogInformation("Processing {DocType} {Date}…", docType, isoDate);

                var (text, sections) = DocumentParser.Parse(html, docType);
                _logger.LogInformation("  → Parsed document: {TextLen} chars, {SectionCount} sections", 
                    text.Length, sections.Count);

                if (string.IsNullOrWhiteSpace(text))
                {
                    _logger.LogWarning("Empty text for {DocType} {Date} — skipping.", docType, isoDate);
                    continue;
                }

                var chunks = DocumentChunker.Chunk(text, docType, isoDate, url, sections);
                _logger.LogInformation("  → Created {Count} chunks", chunks.Count);

                // ── Creative Enhancement: Chunk Enrichment ───────────────────
                // Generate document-level summary (shared context for all chunks)
                _logger.LogInformation("  → Generating document summary...");
                var docSummary = await _enricher.GenerateDocumentSummaryAsync(chunks, ct);
                if (docSummary != null)
                    _logger.LogInformation("  → Document summary: {S}", Truncate(docSummary, 80));
                else
                    _logger.LogInformation("  → No document summary generated (enrichment may be disabled)");

                // Enrich chunks with summaries, entities, key passage flags
                _logger.LogInformation("  → Enriching {Count} chunks with summaries and entities…", chunks.Count);
                var enrichedChunks = await _enricher.EnrichAsync(chunks, docSummary, ct);
                _logger.LogInformation("  → Enrichment complete");

                var keyCount = enrichedChunks.Count(c => c.IsKeyPassage);
                _logger.LogInformation("  → {Key} key passages detected", keyCount);

                // ── Summary-Augmented Embeddings ─────────────────────────────
                // Embed (summary + text) for better semantic recall on high-level queries
                _logger.LogInformation("  → Generating embeddings for {Count} chunks...", enrichedChunks.Count);
                int batchIndex = 0;
                for (int i = 0; i < enrichedChunks.Count; i += EmbedBatchSize)
                {
                    batchIndex++;
                    var batch = enrichedChunks.Skip(i).Take(EmbedBatchSize).ToList();
                    _logger.LogDebug("    → Embedding batch {Batch}: {Count} chunks", batchIndex, batch.Count);
                    
                    // Composite text: summary provides semantic boost, raw text preserves detail
                    var textsToEmbed = batch.Select(c => BuildEmbeddingText(c)).ToList();
                    
                    var vecs  = await _embedder.GetEmbeddingsAsync(textsToEmbed, ct);
                    // Upsert now includes both dense AND sparse vectors automatically
                    await _qdrant.UpsertAsync(batch.Zip(vecs, (c, v) => (c, v)), ct);
                }
                _logger.LogInformation("  → Indexed {Count} chunks to Qdrant (dense + sparse)", enrichedChunks.Count);

                totalChunks += enrichedChunks.Count;
            }
        }

        var count = await _qdrant.CountAsync(ct);
        _logger.LogInformation(
            "Ingestion complete. Chunks indexed={NewChunks}. Total in Qdrant={Total}. Hybrid search ready.",
            totalChunks, count);
    }

    /// <summary>
    /// Builds the text to embed: contextual + summary-augmented for better semantic recall.
    /// 
    /// Strategy (Anthropic's Contextual Retrieval):
    ///   1. Prepend document context (type, date, section) for disambiguation
    ///   2. Include summary for high-level semantic matching
    ///   3. Include raw text for specific detail matching
    /// 
    /// This creates an embedding that captures:
    ///   - Document identity → "FOMC Minutes January 2024" queries match correctly
    ///   - High-level semantics (from summary) → matches conceptual queries
    ///   - Specific details (from raw text) → matches precise queries
    /// 
    /// Based on Anthropic research showing 35-49% retrieval failure reduction
    /// for financial documents (SEC filings) with contextual prefixes.
    /// </summary>
    private static string BuildEmbeddingText(FomcChunk chunk)
    {
        var sb = new System.Text.StringBuilder();
        
        // Contextual prefix (Anthropic's key insight for financial documents)
        sb.AppendLine($"Document: FOMC {FormatDocType(chunk.DocType)} ({chunk.MeetingDate})");
        if (!string.IsNullOrEmpty(chunk.SectionTitle))
            sb.AppendLine($"Section: {chunk.SectionTitle}");
        sb.AppendLine();
        
        // Summary for high-level semantic matching
        if (!string.IsNullOrEmpty(chunk.Summary))
        {
            sb.AppendLine($"Summary: {chunk.Summary}");
            sb.AppendLine();
        }
        
        // Raw text for specific detail matching
        sb.Append(chunk.Text);
        
        return sb.ToString();
    }

    private static string FormatDocType(string docType) => docType switch
    {
        "press_statement" => "Press Statement",
        "minutes" => "Meeting Minutes",
        _ => docType.Replace("_", " ")
    };

    private static string Truncate(string s, int maxLen) =>
        s.Length <= maxLen ? s : s[..(maxLen - 3)] + "…";
}
