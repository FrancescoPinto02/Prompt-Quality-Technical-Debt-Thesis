using System.Collections.Immutable;
using System.Globalization;
using System.Reflection;
using System.Runtime.Loader;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;

public sealed class AnalyzerOutput
{
    [JsonPropertyName("files")]
    public List<FileResult> Files { get; set; } = new();
}

public sealed class FileResult
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("diagnostics")]
    public List<DiagnosticResult> Diagnostics { get; set; } = new();
}

public sealed class DiagnosticResult
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("message")]
    public string? Message { get; set; }

    [JsonPropertyName("severity")]
    public string? Severity { get; set; }

    [JsonPropertyName("line")]
    public int? Line { get; set; }

    [JsonPropertyName("column")]
    public int? Column { get; set; }
}

public sealed class SimpleAnalyzerAssemblyLoader : IAnalyzerAssemblyLoader
{
    private readonly Dictionary<string, string> _pathsByAssemblyName =
        new(StringComparer.OrdinalIgnoreCase);

    public SimpleAnalyzerAssemblyLoader()
    {
        AssemblyLoadContext.Default.Resolving += ResolveAssembly;
    }

    public void AddDependencyLocation(string fullPath)
    {
        try
        {
            if (!File.Exists(fullPath))
            {
                return;
            }

            var assemblyName = AssemblyName.GetAssemblyName(fullPath).Name;

            if (!string.IsNullOrWhiteSpace(assemblyName))
            {
                _pathsByAssemblyName[assemblyName] = fullPath;
            }
        }
        catch
        {
            // Ignore invalid or incompatible assemblies.
        }
    }

    public Assembly LoadFromPath(string fullPath)
    {
        return AssemblyLoadContext.Default.LoadFromAssemblyPath(fullPath);
    }

    private Assembly? ResolveAssembly(AssemblyLoadContext context, AssemblyName assemblyName)
    {
        if (assemblyName.Name is not null &&
            _pathsByAssemblyName.TryGetValue(assemblyName.Name, out var path) &&
            File.Exists(path))
        {
            try
            {
                return context.LoadFromAssemblyPath(path);
            }
            catch
            {
                return null;
            }
        }

        return null;
    }
}

public static class Program
{
    private static readonly string[] AnalyzerPackageIds =
    {
        "microsoft.codeanalysis.netanalyzers",
        "microsoft.codeanalysis.csharp.features"
    };

    private static readonly string[] DiagnosticPrefixesToKeep =
    {
        "CS",
        "CA",
        "IDE"
    };

    private static readonly HashSet<string> IgnoredDiagnosticIds =
        new(StringComparer.OrdinalIgnoreCase)
        {
            // Missing external packages / namespaces / assembly references.
            "CS0234",
            "CS0246",

            // Assembly-level metadata warnings produced by the artificial
            // snippet compilation, not by the generated snippet itself.
            "CA1014",
            "CA1016",
            "CA1017",

            // Hidden compiler diagnostic for unnecessary using directives.
            // We keep IDE0005 instead, which represents the same issue more cleanly.
            "CS8019"
        };

    private static readonly Lazy<List<MetadataReference>> References =
        new(GetMetadataReferences);

    private static readonly Lazy<ImmutableArray<DiagnosticAnalyzer>> AnalyzerCache =
        new(LoadAnalyzers);

    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        var files = ReadInputFiles(args)
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Select(path => System.IO.Path.GetFullPath(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        var output = new AnalyzerOutput();

        foreach (var file in files)
        {
            output.Files.Add(AnalyzeFile(file));
        }

        var json = JsonSerializer.Serialize(
            output,
            new JsonSerializerOptions
            {
                WriteIndented = false,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            }
        );

        Console.WriteLine(json);
        return 0;
    }

    private static List<string> ReadInputFiles(string[] args)
    {
        var files = new List<string>();

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];

            if (arg == "--file-list" && i + 1 < args.Length)
            {
                var fileListPath = args[i + 1];

                if (File.Exists(fileListPath))
                {
                    files.AddRange(
                        File.ReadAllLines(fileListPath)
                            .Select(line => line.Trim())
                            .Where(line => line.Length > 0)
                    );
                }

                i++;
                continue;
            }

            files.Add(arg);
        }

        return files;
    }

    private static FileResult AnalyzeFile(string filePath)
    {
        var result = new FileResult
        {
            Path = filePath
        };

        try
        {
            var code = File.ReadAllText(filePath, Encoding.UTF8);

            var parseOptions = CSharpParseOptions.Default
                .WithLanguageVersion(LanguageVersion.Latest)
                .WithKind(SourceCodeKind.Regular);

            var sourceText = SourceText.From(code, Encoding.UTF8);

            var syntaxTree = CSharpSyntaxTree.ParseText(
                sourceText,
                parseOptions,
                path: filePath
            );

            var analyzers = AnalyzerCache.Value;

            var compilationOptions = new CSharpCompilationOptions(OutputKind.ConsoleApplication)
                .WithNullableContextOptions(NullableContextOptions.Enable)
                .WithSpecificDiagnosticOptions(BuildSpecificDiagnosticOptions(analyzers));

            var compilation = CSharpCompilation.Create(
                assemblyName: "SnippetAssembly" + Guid.NewGuid().ToString("N"),
                syntaxTrees: new[] { syntaxTree },
                references: References.Value,
                options: compilationOptions
            );

            var compilerDiagnostics = compilation.GetDiagnostics();

            var analyzerDiagnostics = ImmutableArray<Diagnostic>.Empty;

            if (analyzers.Length > 0)
            {
                var analyzerOptions = new AnalyzerOptions(
                    ImmutableArray<AdditionalText>.Empty
                );

                analyzerDiagnostics = compilation
                    .WithAnalyzers(analyzers, analyzerOptions)
                    .GetAnalyzerDiagnosticsAsync()
                    .GetAwaiter()
                    .GetResult();
            }

            var diagnostics = compilerDiagnostics
                .Concat(analyzerDiagnostics)
                .Where(ShouldKeepDiagnostic)
                .GroupBy(diagnostic => DiagnosticKey(diagnostic))
                .Select(group => group.First())
                .OrderBy(diagnostic => diagnostic.Location.IsInSource ? diagnostic.Location.GetLineSpan().StartLinePosition.Line : int.MaxValue)
                .ThenBy(diagnostic => diagnostic.Location.IsInSource ? diagnostic.Location.GetLineSpan().StartLinePosition.Character : int.MaxValue)
                .ThenBy(diagnostic => diagnostic.Id)
                .Select(ToDiagnosticResult)
                .ToList();

            result.Diagnostics = diagnostics;
        }
        catch (Exception ex)
        {
            result.Diagnostics.Add(
                new DiagnosticResult
                {
                    Id = "ROSLYN_ANALYZER_ERROR",
                    Message = ex.Message,
                    Severity = "Error",
                    Line = null,
                    Column = null
                }
            );
        }

        return result;
    }

    private static ImmutableDictionary<string, ReportDiagnostic> BuildSpecificDiagnosticOptions(
        ImmutableArray<DiagnosticAnalyzer> analyzers
    )
    {
        var builder = ImmutableDictionary.CreateBuilder<string, ReportDiagnostic>(
            StringComparer.OrdinalIgnoreCase
        );

        foreach (var ignoredId in IgnoredDiagnosticIds)
        {
            builder[ignoredId] = ReportDiagnostic.Suppress;
        }

        foreach (var analyzer in analyzers)
        {
            foreach (var descriptor in analyzer.SupportedDiagnostics)
            {
                if (ShouldKeepDiagnosticId(descriptor.Id) && !IgnoredDiagnosticIds.Contains(descriptor.Id))
                {
                    // Force CA and IDE diagnostics to be emitted.
                    builder[descriptor.Id] = ReportDiagnostic.Warn;
                }
            }
        }

        return builder.ToImmutable();
    }

    private static bool ShouldKeepDiagnostic(Diagnostic diagnostic)
    {
        if (IgnoredDiagnosticIds.Contains(diagnostic.Id))
        {
            return false;
        }

        return ShouldKeepDiagnosticId(diagnostic.Id);
    }

    private static bool ShouldKeepDiagnosticId(string? diagnosticId)
    {
        if (string.IsNullOrWhiteSpace(diagnosticId))
        {
            return false;
        }

        return DiagnosticPrefixesToKeep.Any(
            prefix => diagnosticId.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
        );
    }

    private static string DiagnosticKey(Diagnostic diagnostic)
    {
        int? line = null;
        int? column = null;

        if (diagnostic.Location.IsInSource)
        {
            var lineSpan = diagnostic.Location.GetLineSpan();
            line = lineSpan.StartLinePosition.Line + 1;
            column = lineSpan.StartLinePosition.Character + 1;
        }

        return $"{diagnostic.Id}|{line}|{column}|{diagnostic.GetMessage(CultureInfo.InvariantCulture)}";
    }

    private static DiagnosticResult ToDiagnosticResult(Diagnostic diagnostic)
    {
        int? line = null;
        int? column = null;

        if (diagnostic.Location.IsInSource)
        {
            var lineSpan = diagnostic.Location.GetLineSpan();
            line = lineSpan.StartLinePosition.Line + 1;
            column = lineSpan.StartLinePosition.Character + 1;
        }

        return new DiagnosticResult
        {
            Id = diagnostic.Id,
            Message = diagnostic.GetMessage(CultureInfo.InvariantCulture),
            Severity = diagnostic.Severity.ToString(),
            Line = line,
            Column = column
        };
    }

    private static ImmutableArray<DiagnosticAnalyzer> LoadAnalyzers()
    {
        var analyzerPaths = DiscoverAnalyzerAssemblyPaths();
        var dependencyPaths = DiscoverDependencyAssemblyPaths();

        var loader = new SimpleAnalyzerAssemblyLoader();

        foreach (var dependencyPath in dependencyPaths)
        {
            loader.AddDependencyLocation(dependencyPath);
        }

        var analyzers = new List<DiagnosticAnalyzer>();

        foreach (var analyzerPath in analyzerPaths)
        {
            try
            {
                var reference = new AnalyzerFileReference(analyzerPath, loader);
                analyzers.AddRange(reference.GetAnalyzers(LanguageNames.CSharp));
            }
            catch
            {
                // Ignore analyzer assemblies that cannot be loaded in this environment.
            }
        }

        return analyzers
            .GroupBy(analyzer => analyzer.GetType().FullName)
            .Select(group => group.First())
            .ToImmutableArray();
    }

    private static List<string> DiscoverAnalyzerAssemblyPaths()
    {
        var paths = new List<string>();

        foreach (var packageDir in GetLatestAnalyzerPackageDirs())
        {
            foreach (var dllPath in Directory.GetFiles(packageDir, "*.dll", SearchOption.AllDirectories))
            {
                if (LooksLikeAnalyzerAssembly(dllPath))
                {
                    paths.Add(dllPath);
                }
            }
        }

        return paths
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<string> DiscoverDependencyAssemblyPaths()
    {
        var paths = new List<string>();

        if (Directory.Exists(AppContext.BaseDirectory))
        {
            paths.AddRange(
                Directory.GetFiles(AppContext.BaseDirectory, "*.dll", SearchOption.TopDirectoryOnly)
            );
        }

        foreach (var packageDir in GetLatestAnalyzerPackageDirs())
        {
            paths.AddRange(
                Directory.GetFiles(packageDir, "*.dll", SearchOption.AllDirectories)
            );
        }

        return paths
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static bool LooksLikeAnalyzerAssembly(string dllPath)
    {
        var normalized = dllPath.Replace("\\", "/").ToLowerInvariant();
        var filename = System.IO.Path.GetFileName(dllPath);

        if (normalized.Contains("/analyzers/"))
        {
            return true;
        }

        if (filename.Contains("CodeStyle", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (filename.Equals("Microsoft.CodeAnalysis.CSharp.Features.dll", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (filename.Equals("Microsoft.CodeAnalysis.Features.dll", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return false;
    }

    private static List<string> GetLatestAnalyzerPackageDirs()
    {
        var packageRoot = GetNuGetPackageRoot();
        var packageDirs = new List<string>();

        foreach (var packageId in AnalyzerPackageIds)
        {
            var packageBaseDir = System.IO.Path.Combine(packageRoot, packageId);

            if (!Directory.Exists(packageBaseDir))
            {
                continue;
            }

            var latestVersionDir = Directory
                .GetDirectories(packageBaseDir)
                .OrderByDescending(System.IO.Path.GetFileName, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();

            if (latestVersionDir is not null)
            {
                packageDirs.Add(latestVersionDir);
            }
        }

        return packageDirs;
    }

    private static string GetNuGetPackageRoot()
    {
        var fromEnvironment = Environment.GetEnvironmentVariable("NUGET_PACKAGES");

        if (!string.IsNullOrWhiteSpace(fromEnvironment))
        {
            return fromEnvironment;
        }

        return System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".nuget",
            "packages"
        );
    }

    private static List<MetadataReference> GetMetadataReferences()
    {
        var references = new List<MetadataReference>();

        var trustedPlatformAssemblies =
            AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string;

        if (!string.IsNullOrWhiteSpace(trustedPlatformAssemblies))
        {
            foreach (var path in trustedPlatformAssemblies.Split(System.IO.Path.PathSeparator))
            {
                if (File.Exists(path))
                {
                    references.Add(MetadataReference.CreateFromFile(path));
                }
            }

            return references;
        }

        var fallbackAssemblies = new[]
        {
            typeof(object).Assembly.Location,
            typeof(Console).Assembly.Location,
            typeof(Enumerable).Assembly.Location
        };

        foreach (var path in fallbackAssemblies)
        {
            if (File.Exists(path))
            {
                references.Add(MetadataReference.CreateFromFile(path));
            }
        }

        return references;
    }
}
