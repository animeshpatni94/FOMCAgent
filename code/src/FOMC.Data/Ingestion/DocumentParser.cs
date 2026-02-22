using HtmlAgilityPack;
using System.Text;
using System.Text.RegularExpressions;

namespace FOMC.Data.Ingestion;

/// <summary>
/// Converts Fed Reserve HTML pages into clean, structured plain-text.
///
/// Strategy:
///  1. Remove boilerplate nodes (nav, header, footer, script, style).
///  2. For press statements: extract the main article div.
///  3. For minutes: extract the content area and preserve section headings.
///  4. Normalise whitespace and strip residual HTML entities.
/// </summary>
public static class DocumentParser
{
    // Section headings we want to preserve as metadata hints for the chunker.
    private static readonly Regex SectionHeadingPattern =
        new(@"^(Economic\s+Activity|Labor\s+Market|Inflation|Financial\s|Committee\s|Participants'|Staff\s|Monetary\s+Policy|Voting|Open\s+Market|Supply\s+and\s+Demand)",
            RegexOptions.IgnoreCase | RegexOptions.Multiline);

    private static readonly string[] RemoveSelectors =
        ["//nav", "//header", "//footer", "//script", "//style",
         "//*[@id='skip-to-content']", "//*[contains(@class,'utility-bar')]",
         "//*[contains(@class,'breadcrumb')]", "//*[contains(@class,'pagination')]"];

    /// <summary>
    /// Parse HTML → (cleanText, sectionTitles list).
    /// sectionTitles is ordered list of H2/H3 headings found in the document.
    /// </summary>
    public static (string Text, List<string> Sections) Parse(string html, string docType)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        // Remove boilerplate
        foreach (var selector in RemoveSelectors)
        {
            var nodes = doc.DocumentNode.SelectNodes(selector);
            if (nodes is null) continue;
            foreach (var n in nodes.ToList()) n.Remove();
        }

        // Extract main content node
        var contentNode = docType == "minutes"
            ? doc.DocumentNode.SelectSingleNode(
                  "//*[@id='article']"
                  ?? "//*[contains(@class,'col-xs-12 col-sm-8')]")
              ?? doc.DocumentNode.SelectSingleNode("//article")
              ?? doc.DocumentNode
            : doc.DocumentNode.SelectSingleNode(
                  "//div[contains(@class,'col-xs-12')]")
              ?? doc.DocumentNode.SelectSingleNode("//article")
              ?? doc.DocumentNode;

        // Collect section headings (h2, h3)
        var sections = new List<string>();
        var headings = contentNode.SelectNodes(".//h2 | .//h3");
        if (headings is not null)
            foreach (var h in headings)
                sections.Add(CleanText(h.InnerText));

        // Build plain text, inserting heading markers for the chunker
        var sb = new StringBuilder();
        WalkNode(contentNode, sb);

        var text = CleanText(sb.ToString());
        return (text, sections);
    }

    private static void WalkNode(HtmlNode node, StringBuilder sb)
    {
        switch (node.NodeType)
        {
            case HtmlNodeType.Text:
                var t = HtmlEntity.DeEntitize(node.InnerText);
                if (!string.IsNullOrWhiteSpace(t)) sb.Append(t);
                break;

            case HtmlNodeType.Element:
                var tag = node.Name.ToLowerInvariant();

                // Block elements get newlines around them
                if (tag is "p" or "div" or "li" or "td")
                    sb.Append('\n');

                // Headings get a marker that the chunker uses for splitting
                if (tag is "h1" or "h2" or "h3" or "h4")
                    sb.Append("\n## ");

                foreach (var child in node.ChildNodes)
                    WalkNode(child, sb);

                if (tag is "p" or "div" or "li" or "tr" or "h1" or "h2" or "h3" or "h4")
                    sb.Append('\n');
                break;
        }
    }

    private static string CleanText(string raw)
    {
        // Normalise whitespace
        var s = Regex.Replace(raw, @"[ \t]+", " ");
        // Collapse 3+ newlines into 2
        s = Regex.Replace(s, @"\n{3,}", "\n\n");
        return s.Trim();
    }
}
