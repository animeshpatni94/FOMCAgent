namespace FOMC.Shared.Constants;

/// <summary>
/// FOMC document type constants.
/// Centralizes magic strings for document types.
/// </summary>
public static class FomcDocumentTypes
{
    public const string PressStatement = "press_statement";
    public const string Minutes = "minutes";

    public static readonly IReadOnlyList<string> All = [PressStatement, Minutes];

    public static bool IsValid(string? docType) =>
        docType is null || All.Contains(docType);
}
