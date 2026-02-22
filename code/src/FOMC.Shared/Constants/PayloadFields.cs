namespace FOMC.Shared.Constants;

/// <summary>
/// Payload field names for vector store.
/// Eliminates magic strings in Qdrant operations.
/// </summary>
public static class PayloadFields
{
    public const string ChunkId = "chunk_id";
    public const string DocType = "doc_type";
    public const string MeetingDate = "meeting_date";
    public const string SourceUrl = "source_url";
    public const string Text = "text";
    public const string ChunkIndex = "chunk_index";
    public const string TotalChunks = "total_chunks";
    public const string SectionTitle = "section_title";
    
    // Enrichment fields
    public const string Summary = "summary";
    public const string DocumentSummary = "document_summary";
    public const string IsKeyPassage = "is_key_passage";
    public const string KeyEntities = "key_entities";
}
