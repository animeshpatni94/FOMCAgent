using FOMC.Shared.Abstractions;
using FOMC.Shared.Configuration;
using FOMC.Shared.Constants;
using FOMC.Shared.Models;
using Google.Protobuf.Collections;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace FOMC.Data.Services;

/// <summary>
/// Qdrant vector database service implementing IVectorStore.
/// Supports both dense vectors (HNSW) and sparse vectors (BM25) for hybrid search.
/// </summary>
public sealed class QdrantService : IVectorStore
{
    private readonly QdrantClient _client;
    private readonly QdrantOptions _options;
    private readonly ILogger<QdrantService> _logger;
    private readonly VocabularyService _vocabulary;

    // Named vector configurations
    private const string DenseVectorName = "dense";
    private const string SparseVectorName = "sparse";

    public QdrantService(
        IOptions<QdrantOptions> options, 
        ILogger<QdrantService> logger,
        VocabularyService vocabulary)
    {
        _logger = logger;
        _options = options.Value;
        _client = new QdrantClient(_options.Host, _options.GrpcPort);
        _vocabulary = vocabulary;
        
        _logger.LogInformation(
            "QdrantService {Host}:{Port} collection={Collection}",
            _options.Host, _options.GrpcPort, _options.CollectionName);
    }

    public async Task EnsureCollectionAsync(int vectorSize, CancellationToken ct = default)
    {
        // Load vocabulary for sparse vectors
        await _vocabulary.LoadAsync(ct);
        
        if (await _client.CollectionExistsAsync(_options.CollectionName, ct))
        {
            _logger.LogInformation("Collection '{C}' exists.", _options.CollectionName);
            return;
        }
        
        _logger.LogInformation(
            "Creating collection '{C}' with dense ({S} dims) + sparse vectors, m={M} ef_construct={Ef}",
            _options.CollectionName, vectorSize, _options.HnswM, _options.HnswEfConstruct);

        // Create collection with named vectors: "dense" (HNSW) and "sparse" (BM25)
        var vectorsConfig = new VectorParamsMap();
        vectorsConfig.Map.Add(DenseVectorName, new VectorParams
        {
            Size = (ulong)vectorSize,
            Distance = Distance.Cosine,
            HnswConfig = new HnswConfigDiff
            {
                M = (ulong)_options.HnswM,
                EfConstruct = (ulong)_options.HnswEfConstruct,
                OnDisk = false
            }
        });

        var sparseConfig = new SparseVectorConfig();
        sparseConfig.Map.Add(SparseVectorName, new SparseVectorParams
        {
            Modifier = Modifier.Idf  // Apply IDF weighting at query time
        });

        await _client.CreateCollectionAsync(
            _options.CollectionName,
            vectorsConfig: vectorsConfig,
            sparseVectorsConfig: sparseConfig,
            cancellationToken: ct);
        
        _logger.LogInformation("Collection '{C}' created with hybrid vector support.", _options.CollectionName);
    }

    public async Task DeleteCollectionAsync(CancellationToken ct = default) =>
        await _client.DeleteCollectionAsync(_options.CollectionName, cancellationToken: ct);

    public async Task UpsertAsync(
        IEnumerable<(FomcChunk Chunk, float[] Vector)> items,
        CancellationToken ct = default)
    {
        var points = items.Select(item =>
        {
            // Compute sparse vector for BM25 (updates vocabulary)
            var textForSparse = $"{item.Chunk.Text} {item.Chunk.Summary ?? ""}";
            var (sparseIndices, sparseValues) = _vocabulary.ComputeSparseVector(textForSparse, updateVocab: true);
            
            var p = new PointStruct
            {
                Id = new PointId { Uuid = item.Chunk.ChunkId },
            };
            
            // Named vectors: dense + sparse
            p.Vectors = new Vectors
            {
                Vectors_ = new NamedVectors()
            };
            p.Vectors.Vectors_.Vectors.Add(DenseVectorName, item.Vector);
            
            // Add sparse vector if we have terms
            if (sparseIndices.Length > 0)
            {
                var sparseVec = new Vector
                {
                    Indices = new SparseIndices()
                };
                sparseVec.Indices.Data.AddRange(sparseIndices);
                sparseVec.Data.AddRange(sparseValues);
                p.Vectors.Vectors_.Vectors.Add(SparseVectorName, sparseVec);
            }
            
            // Use constants for payload field names (DRY, refactor-safe)
            p.Payload[PayloadFields.ChunkId] = item.Chunk.ChunkId;
            p.Payload[PayloadFields.DocType] = item.Chunk.DocType;
            p.Payload[PayloadFields.MeetingDate] = item.Chunk.MeetingDate;
            p.Payload[PayloadFields.SourceUrl] = item.Chunk.SourceUrl;
            p.Payload[PayloadFields.Text] = item.Chunk.Text;
            p.Payload[PayloadFields.ChunkIndex] = (long)item.Chunk.ChunkIndex;
            p.Payload[PayloadFields.TotalChunks] = (long)item.Chunk.TotalChunks;
            p.Payload[PayloadFields.SectionTitle] = item.Chunk.SectionTitle ?? "";
            
            // Creative enrichment fields
            p.Payload[PayloadFields.Summary] = item.Chunk.Summary ?? "";
            p.Payload[PayloadFields.DocumentSummary] = item.Chunk.DocumentSummary ?? "";
            p.Payload[PayloadFields.IsKeyPassage] = item.Chunk.IsKeyPassage;
            p.Payload[PayloadFields.KeyEntities] = string.Join("|", item.Chunk.KeyEntities ?? []);
            return p;
        }).ToList();

        await _client.UpsertAsync(_options.CollectionName, points, cancellationToken: ct);
        
        // Save vocabulary after each batch (incremental persistence)
        await _vocabulary.SaveAsync(ct);
        
        _logger.LogDebug("Upserted {N} points with dense + sparse vectors", points.Count);
    }

    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        float[] queryVector,
        int topK = 5,
        string? docType = null,
        string? dateFrom = null,
        string? dateTo = null,
        int hnswEf = 64,
        bool exact = false,
        CancellationToken ct = default)
    {
        var filter = BuildFilter(docType);

        var searchParams = exact
            ? new SearchParams { Exact = true }
            : new SearchParams { HnswEf = (ulong)hnswEf };

        // Use named vector for dense search
        var hits = await _client.SearchAsync(
            _options.CollectionName,
            queryVector,
            vectorName: DenseVectorName,
            filter: filter,
            limit: (ulong)topK,
            payloadSelector: true,
            searchParams: searchParams,
            cancellationToken: ct);

        var results = hits.Select(MapToResult).ToList();

        // Apply date filters (post-filter since Qdrant doesn't support date range natively)
        if (dateFrom is not null)
            results = results.Where(r => string.Compare(r.Chunk.MeetingDate, dateFrom, StringComparison.Ordinal) >= 0).ToList();
        if (dateTo is not null)
            results = results.Where(r => string.Compare(r.Chunk.MeetingDate, dateTo, StringComparison.Ordinal) <= 0).ToList();

        return results;
    }

    /// <summary>
    /// Hybrid search using Qdrant's native Query API with prefetch + RRF fusion.
    /// Combines dense vectors (semantic) with sparse vectors (BM25 keywords).
    /// 
    /// Based on Anthropic's Contextual Retrieval research:
    /// - Dense search excels at semantic similarity ("Fed's stance on inflation")
    /// - Keyword search excels at exact matches ("5.25%", "March 2024", "FOMC")
    /// - Combined: 49% fewer retrieval failures on financial documents
    /// </summary>
    public async Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        float[] queryVector,
        string queryText,
        int topK = 5,
        float denseWeight = 0.7f,
        string? docType = null,
        string? dateFrom = null,
        string? dateTo = null,
        CancellationToken ct = default)
    {
        var filter = BuildFilter(docType);
        
        // Compute sparse query vector from text
        var (sparseIndices, sparseValues) = _vocabulary.ComputeSparseVector(queryText, updateVocab: false);
        
        // If no sparse terms, fall back to dense-only search
        if (sparseIndices.Length == 0)
        {
            _logger.LogDebug("No sparse terms for query, using dense-only search");
            return await SearchAsync(queryVector, topK, docType, dateFrom, dateTo, ct: ct);
        }

        // Use Qdrant Query API with prefetch for hybrid search
        // Prefetch retrieves candidates from both dense and sparse indexes
        // Then RRF fusion combines them
        var candidateCount = (ulong)(topK * 3);
        
        try
        {
            // Build dense vector input
            var denseVector = new DenseVector();
            denseVector.Data.AddRange(queryVector);
            
            // Build sparse vector input
            var sparseVector = new SparseVector();
            sparseVector.Indices.AddRange(sparseIndices);
            sparseVector.Values.AddRange(sparseValues);
            
            // Build prefetch queries for both dense and sparse
            var prefetchQueries = new List<PrefetchQuery>
            {
                // Dense (semantic) search
                new PrefetchQuery
                {
                    Query = new Query { Nearest = new VectorInput { Dense = denseVector } },
                    Using = DenseVectorName,
                    Limit = candidateCount
                },
                // Sparse (BM25) search
                new PrefetchQuery
                {
                    Query = new Query 
                    { 
                        Nearest = new VectorInput { Sparse = sparseVector }
                    },
                    Using = SparseVectorName,
                    Limit = candidateCount
                }
            };

            // Execute hybrid query with RRF fusion
            var hits = await _client.QueryAsync(
                _options.CollectionName,
                query: new Query { Fusion = Fusion.Rrf },
                prefetch: prefetchQueries,
                filter: filter,
                limit: (ulong)topK,
                payloadSelector: true,
                cancellationToken: ct);

            var results = hits.Select(MapScoredPointToResult).ToList();

            // Apply date filters
            if (dateFrom is not null)
                results = results.Where(r => string.Compare(r.Chunk.MeetingDate, dateFrom, StringComparison.Ordinal) >= 0).ToList();
            if (dateTo is not null)
                results = results.Where(r => string.Compare(r.Chunk.MeetingDate, dateTo, StringComparison.Ordinal) <= 0).ToList();

            _logger.LogDebug("HybridSearch: returned {Count} results using Qdrant RRF fusion", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Hybrid search failed, falling back to dense-only search");
            return await SearchAsync(queryVector, topK, docType, dateFrom, dateTo, ct: ct);
        }
    }

    /// <summary>
    /// Build a Qdrant filter for document type.
    /// </summary>
    private static Filter? BuildFilter(string? docType)
    {
        if (docType is null) return null;
        
        return new Filter
        {
            Must =
            {
                new Condition
                {
                    Field = new FieldCondition
                    {
                        Key = PayloadFields.DocType,
                        Match = new Match { Keyword = docType }
                    }
                }
            }
        };
    }

    public async Task<long> CountAsync(CancellationToken ct = default)
    {
        var info = await _client.GetCollectionInfoAsync(_options.CollectionName, ct);
        return (long)info.PointsCount;
    }

    public async Task<List<string>> GetMeetingDatesAsync(CancellationToken ct = default)
    {
        var dates = new SortedSet<string>();
        PointId? offset = null;

        while (true)
        {
            var scroll = await _client.ScrollAsync(
                _options.CollectionName,
                limit: 250,
                offset: offset,
                payloadSelector: new WithPayloadSelector { Enable = true },
                vectorsSelector: new WithVectorsSelector { Enable = false },
                cancellationToken: ct);

            foreach (var pt in scroll.Result)
                if (pt.Payload.TryGetValue(PayloadFields.MeetingDate, out var v))
                    dates.Add(v.StringValue);

            if (scroll.NextPageOffset is null || !scroll.Result.Any()) break;
            offset = scroll.NextPageOffset;
        }

        return [.. dates];
    }

    public async Task<FomcChunk?> GetChunkAsync(string chunkId, CancellationToken ct = default)
    {
        var pts = await _client.RetrieveAsync(
            _options.CollectionName,
            [new PointId { Uuid = chunkId }],
            withPayload: true,
            withVectors: false,
            cancellationToken: ct);

        return pts.FirstOrDefault() is { } p ? MapToChunk(p.Payload) : null;
    }

    // ── Mapping helpers using PayloadFields constants ────────────────────────

    private static SearchResult MapToResult(ScoredPoint p) => new()
    {
        Score = p.Score,
        Chunk = MapToChunk(p.Payload)
    };

    private static SearchResult MapScoredPointToResult(ScoredPoint p) => new()
    {
        Score = p.Score,
        Chunk = MapToChunk(p.Payload)
    };

    private static FomcChunk MapToChunk(MapField<string, Value> payload) => new()
    {
        ChunkId = payload[PayloadFields.ChunkId].StringValue,
        DocType = payload[PayloadFields.DocType].StringValue,
        MeetingDate = payload[PayloadFields.MeetingDate].StringValue,
        SourceUrl = payload[PayloadFields.SourceUrl].StringValue,
        Text = payload[PayloadFields.Text].StringValue,
        ChunkIndex = (int)payload[PayloadFields.ChunkIndex].IntegerValue,
        TotalChunks = (int)payload[PayloadFields.TotalChunks].IntegerValue,
        SectionTitle = GetOptionalString(payload, PayloadFields.SectionTitle),
        Summary = GetOptionalString(payload, PayloadFields.Summary),
        DocumentSummary = GetOptionalString(payload, PayloadFields.DocumentSummary),
        IsKeyPassage = payload.TryGetValue(PayloadFields.IsKeyPassage, out var kp) && kp.BoolValue,
        KeyEntities = GetOptionalStringList(payload, PayloadFields.KeyEntities)
    };

    private static string? GetOptionalString(MapField<string, Value> payload, string key) =>
        payload.TryGetValue(key, out var v) && !string.IsNullOrEmpty(v.StringValue)
            ? v.StringValue
            : null;

    private static List<string>? GetOptionalStringList(MapField<string, Value> payload, string key) =>
        payload.TryGetValue(key, out var v) && !string.IsNullOrEmpty(v.StringValue)
            ? v.StringValue.Split('|', StringSplitOptions.RemoveEmptyEntries).ToList()
            : null;
}
