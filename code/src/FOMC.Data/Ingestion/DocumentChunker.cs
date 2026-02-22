using FOMC.Shared.Models;
using System.Security.Cryptography;
using System.Text;

namespace FOMC.Data.Ingestion;

/// <summary>
/// Splits cleaned document text into overlapping chunks with metadata.
///
/// Design choices:
///  - Recursive character splitting: tries paragraph → sentence → word boundaries
///    to avoid cutting mid-thought.
///  - Chunk sizes differ by document type:
///      Press statements: 400 chars / 60 overlap  (short, dense policy language)
///      Minutes:          700 chars / 100 overlap  (long analytical prose)
///  - Section titles are forwarded from the parser for richer citation context.
///  - ChunkId is a deterministic UUID from (docType + date + chunkIndex) so
///    re-ingestion is idempotent.
/// </summary>
public static class DocumentChunker
{
    private static readonly string[] Separators =
        ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "];

    /// <summary>
    /// Chunk a document and return FomcChunk records ready for embedding.
    /// </summary>
    public static List<FomcChunk> Chunk(
        string   text,
        string   docType,
        string   meetingDate, // "YYYY-MM-DD"
        string   sourceUrl,
        List<string>? sectionTitles = null)
    {
        var (chunkSize, overlap) = docType == "press_statement"
            ? (400, 60)
            : (700, 100);

        var rawChunks = SplitText(text, chunkSize, overlap);

        var results = new List<FomcChunk>(rawChunks.Count);
        for (int i = 0; i < rawChunks.Count; i++)
        {
            var chunkText = rawChunks[i];
            // Assign the nearest section title that appears before this chunk
            var sectionTitle = FindSection(sectionTitles, text, chunkText);

            results.Add(new FomcChunk
            {
                ChunkId      = MakeDeterministicId(docType, meetingDate, i),
                DocType      = docType,
                MeetingDate  = meetingDate,
                SourceUrl    = sourceUrl,
                Text         = chunkText,
                ChunkIndex   = i,
                TotalChunks  = rawChunks.Count,
                SectionTitle = sectionTitle
            });
        }

        return results;
    }

    // ── Core splitting logic ─────────────────────────────────────────────────

    private static List<string> SplitText(string text, int maxChars, int overlapChars)
    {
        var chunks  = new List<string>();
        int start   = 0;

        while (start < text.Length)
        {
            int end = Math.Min(start + maxChars, text.Length);

            // If we're not at the end of the text, walk back to a good break
            if (end < text.Length)
            {
                int breakAt = FindBreakPoint(text, start, end);
                if (breakAt > start) end = breakAt;
            }

            var chunk = text[start..end].Trim();
            if (chunk.Length > 20) // skip near-empty fragments
                chunks.Add(chunk);

            // Advance with overlap to preserve context across chunk boundaries
            int nextStart = end - overlapChars;
            if (nextStart <= start) nextStart = start + 1; // safety guard
            start = nextStart;
        }

        return chunks;
    }

    /// <summary>
    /// Find the latest separator position in (start, end] that is at least
    /// halfway through the chunk, preferring paragraph > sentence > word breaks.
    /// </summary>
    private static int FindBreakPoint(string text, int start, int end)
    {
        int midpoint = start + (end - start) / 2;

        foreach (var sep in Separators)
        {
            int pos = text.LastIndexOf(sep, end - 1, end - midpoint);
            if (pos > midpoint)
                return pos + sep.Length;
        }

        return end; // fallback: hard cut
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static string? FindSection(List<string>? sections, string fullText, string chunkText)
    {
        if (sections is null || sections.Count == 0) return null;
        int chunkPos = fullText.IndexOf(chunkText, StringComparison.Ordinal);
        if (chunkPos < 0) return null;

        // Find the last section heading that starts before this chunk
        string? best = null;
        foreach (var s in sections)
        {
            int sPos = fullText.IndexOf(s, StringComparison.Ordinal);
            if (sPos >= 0 && sPos <= chunkPos)
                best = s;
        }
        return best;
    }

    /// <summary>
    /// Create a stable UUID from doc identity so that re-running ingestion
    /// overwrites rather than duplicates entries in Qdrant.
    /// </summary>
    private static string MakeDeterministicId(string docType, string date, int index)
    {
        var key   = $"{docType}|{date}|{index}";
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(key));
        // Take first 16 bytes and format as a UUID v4-style string
        bytes[6] = (byte)((bytes[6] & 0x0F) | 0x40);
        bytes[8] = (byte)((bytes[8] & 0x3F) | 0x80);
        return new Guid(bytes[..16]).ToString();
    }
}
