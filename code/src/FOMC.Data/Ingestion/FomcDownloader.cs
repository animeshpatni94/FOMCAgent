using Microsoft.Extensions.Logging;

namespace FOMC.Data.Ingestion;

/// <summary>
/// Downloads raw HTML for FOMC press statements and meeting minutes
/// from federalreserve.gov.  Stores them to a local cache directory
/// so re-running ingestion doesn't hit the network again.
/// </summary>
public class FomcDownloader
{
    private readonly HttpClient _http;
    private readonly string     _cacheDir;
    private readonly ILogger<FomcDownloader> _logger;

    // Complete list of FOMC meeting dates 2024-2025.
    // Format: YYYYMMDD — matches the Fed's URL scheme.
    public static readonly string[] MeetingDates =
    [
        // 2024
        "20240131", "20240320", "20240501", "20240612",
        "20240731", "20240918", "20241107", "20241218",
        // 2025
        "20250129", "20250319", "20250507", "20250618",
        "20250730", "20250917", "20251105", "20251217"
    ];

    // URL templates for the two document types
    private const string StatementUrlTemplate =
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary{0}a.htm";
    private const string MinutesUrlTemplate =
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes{0}.htm";

    public FomcDownloader(HttpClient http, string cacheDir, ILogger<FomcDownloader> logger)
    {
        _http     = http;
        _cacheDir = cacheDir;
        _logger   = logger;
        Directory.CreateDirectory(cacheDir);
    }

    /// <summary>
    /// Download all documents for a given date.
    /// Returns (docType, date, url, html) tuples for documents that exist.
    /// </summary>
    public async Task<List<(string DocType, string Date, string Url, string Html)>>
        DownloadMeetingAsync(string date, CancellationToken ct = default)
    {
        var results = new List<(string, string, string, string)>();

        var docs = new[]
        {
            ("press_statement", string.Format(StatementUrlTemplate, date)),
            ("minutes",         string.Format(MinutesUrlTemplate,   date))
        };

        foreach (var (docType, url) in docs)
        {
            var cacheFile = Path.Combine(_cacheDir, $"{date}_{docType}.html");

            string html;
            if (File.Exists(cacheFile))
            {
                _logger.LogDebug("Cache hit: {File}", cacheFile);
                html = await File.ReadAllTextAsync(cacheFile, ct);
            }
            else
            {
                _logger.LogInformation("Downloading {DocType} for {Date}…", docType, date);
                try
                {
                    html = await _http.GetStringAsync(url, ct);
                    await File.WriteAllTextAsync(cacheFile, html, ct);
                }
                catch (HttpRequestException ex)
                {
                    // Some dates may not have all document types (e.g. emergency meetings).
                    _logger.LogWarning("Skipping {Url}: {Error}", url, ex.Message);
                    continue;
                }
            }

            results.Add((docType, date, url, html));
        }

        return results;
    }

    /// <summary>Download all meetings; returns aggregated document list.</summary>
    public async Task<List<(string DocType, string Date, string Url, string Html)>>
        DownloadAllAsync(CancellationToken ct = default)
    {
        var all = new List<(string, string, string, string)>();
        foreach (var date in MeetingDates)
        {
            var docs = await DownloadMeetingAsync(date, ct);
            all.AddRange(docs);
            await Task.Delay(300, ct); // polite crawl rate
        }
        return all;
    }
}
