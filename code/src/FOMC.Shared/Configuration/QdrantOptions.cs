namespace FOMC.Shared.Configuration;

/// <summary>
/// Strongly-typed configuration for Qdrant vector database.
/// Bind from appsettings.json section "Qdrant".
/// </summary>
public sealed class QdrantOptions
{
    public const string SectionName = "Qdrant";

    public string Host { get; set; } = "localhost";

    public int GrpcPort { get; set; } = 6334;

    public int HttpPort { get; set; } = 6333;

    public string CollectionName { get; set; } = "fomc_documents";

    /// <summary>
    /// HNSW index parameter: number of bi-directional links per node.
    /// Higher = better recall, more memory. Typical: 12-48.
    /// </summary>
    public int HnswM { get; set; } = 16;

    /// <summary>
    /// HNSW index parameter: size of dynamic candidate list during construction.
    /// Higher = better quality, slower build. Typical: 64-512.
    /// </summary>
    public int HnswEfConstruct { get; set; } = 100;

    /// <summary>
    /// Default ef parameter for searches. Higher = better recall, slower search.
    /// </summary>
    public int DefaultSearchEf { get; set; } = 64;

    /// <summary>
    /// Weight for dense (semantic) vectors in hybrid search (0.0 to 1.0).
    /// BM25 weight = 1 - DenseWeight. Default 0.7 means 70% semantic, 30% keyword.
    /// </summary>
    public float HybridDenseWeight { get; set; } = 0.7f;
}
