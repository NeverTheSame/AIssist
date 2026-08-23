package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Finding is one static-analysis hit. Field names/json tags are stable so
// --format=json output can be diffed or piped into another tool.
type Finding struct {
	Check   string `json:"check"`
	File    string `json:"file"`
	Line    int    `json:"line"`
	Message string `json:"message"`
}

var skipDirNames = map[string]bool{
	".git":          true,
	"venv":          true,
	"__pycache__":   true,
	"node_modules":  true,
	".pytest_cache": true,
	".guard_audit":  true,
	"logs":          true,
	".idea":         true,
	".vscode":       true,
}

func walkPythonFiles(root string) ([]string, error) {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			base := info.Name()
			if skipDirNames[base] || strings.HasPrefix(base, "venv") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".py") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

func relPath(root, path string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return path
	}
	return filepath.ToSlash(rel)
}

// --- Check 1: any AzureOpenAI(...) construction that isn't the one
// sanctioned factory (azure_auth.get_openai_client_with_auth), which is
// the only place a client leaves wrapped by guard.wrap_client. A client
// built anywhere else never gets its chat.completions.create redacted. ---

var azureOpenAIConstructRe = regexp.MustCompile(`\bAzureOpenAI\s*\(`)

func checkBypassedGateway(root string, files []string) ([]Finding, error) {
	var findings []Finding
	sanctionedFile := filepath.Join(root, "azure_auth.py")

	for _, file := range files {
		if samePath(file, sanctionedFile) {
			continue
		}
		lines, err := readLines(file)
		if err != nil {
			return nil, err
		}
		for i, line := range lines {
			if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
				continue
			}
			if azureOpenAIConstructRe.MatchString(line) {
				findings = append(findings, Finding{
					Check: "bypassed-gateway",
					File:  relPath(root, file),
					Line:  i + 1,
					Message: "AzureOpenAI(...) constructed outside azure_auth.get_openai_client_with_auth; " +
						"this client will never pass through guard.wrap_client, so its " +
						"chat.completions.create calls bypass the redaction gateway entirely",
				})
			}
		}
	}
	return findings, nil
}

func samePath(a, b string) bool {
	absA, errA := filepath.Abs(a)
	absB, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return a == b
	}
	return absA == absB
}

// --- Check 2: print() calls that interpolate a known-sensitive variable.
// guard.install_log_redaction() only attaches to logging handlers -- a
// bare print() bypasses the logging module (and therefore the redaction
// filter) completely, so this is a real, separate leak path. ---

var printCallRe = regexp.MustCompile(`\bprint\s*\(`)

// Deliberately narrow: names unambiguously bound to raw incident content or
// a fully-assembled prompt on its way to the model or a log. Container/
// result variable names (incident_data, summary_result) are excluded --
// they're also the analyst-facing rehydrated output this whole layer
// exists to let the analyst read, so flagging them would just be noise.
var sensitiveVarNames = []string{
	"system_prompt",
	"user_prompt",
	"enhanced_user_prompt",
	"full_content",
	"raw_content",
	"teams_discussion",
	"conversation_text",
	"analysis_text",
}

func checkUnredactedPrint(root string, files []string) ([]Finding, error) {
	var findings []Finding
	varRes := make([]*regexp.Regexp, len(sensitiveVarNames))
	bareArgRes := make([]*regexp.Regexp, len(sensitiveVarNames))
	for i, name := range sensitiveVarNames {
		varRes[i] = regexp.MustCompile(`\b` + regexp.QuoteMeta(name) + `\b`)
		bareArgRes[i] = regexp.MustCompile(`print\(\s*` + regexp.QuoteMeta(name) + `\b`)
	}

	for _, file := range files {
		lines, err := readLines(file)
		if err != nil {
			return nil, err
		}
		for i, line := range lines {
			if !printCallRe.MatchString(line) {
				continue
			}
			if strings.Contains(line, "guard.redact_text(") {
				continue // already scrubbed inline before printing
			}
			// Only count a hit inside an f-string interpolation ({...}) or
			// as a bare/attribute argument right after "print(" -- a var
			// name that merely appears inside a plain string literal (e.g.
			// a "--- system_prompt ---" section header) isn't interpolated
			// content and would just be noise.
			for j, name := range sensitiveVarNames {
				matched := bareArgRes[j].MatchString(line)
				if !matched {
					for _, brace := range braceExprRe.FindAllString(line, -1) {
						if varRes[j].MatchString(brace) {
							matched = true
							break
						}
					}
				}
				if matched {
					findings = append(findings, Finding{
						Check: "unredacted-print",
						File:  relPath(root, file),
						Line:  i + 1,
						Message: fmt.Sprintf(
							"print() interpolates %q without guard.redact_text(); print() bypasses "+
								"guard.install_log_redaction, which only filters logging handlers",
							name),
					})
					break
				}
			}
		}
	}
	return findings, nil
}

// --- Check 3: file writes to a literal path not covered by .gitignore.
// Catches the next `error.log`-style mistake -- a new debug/diagnostic
// file that can contain incident content and gets committed by accident. ---

// Go's RE2 engine has no backreferences, so this doesn't enforce that the
// opening/closing quote characters match -- an acceptable miss for a
// heuristic checker (Python code mixing quote styles on one open() call
// would be unusual style regardless).
var openWriteRe = regexp.MustCompile(`open\(\s*f?["']((?:[^"'{}]|\{[^}]*\})*)["']\s*,\s*["'](w|a)b?["']`)
var braceExprRe = regexp.MustCompile(`\{[^}]*\}`)

func checkUngitignoredWrites(root string, files []string) ([]Finding, error) {
	var findings []Finding

	for _, file := range files {
		lines, err := readLines(file)
		if err != nil {
			return nil, err
		}
		for i, line := range lines {
			m := openWriteRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			literal, mode := m[1], m[2]
			resolved := braceExprRe.ReplaceAllString(literal, "x")
			if resolved == "" || strings.HasPrefix(resolved, "/") || strings.Contains(resolved, "..") {
				continue // dynamic-only or absolute/traversal path; best-effort check, skip rather than guess
			}

			ignored, err := isGitIgnored(root, resolved)
			if err != nil {
				continue // git unavailable in this environment; don't false-flag
			}
			if !ignored {
				findings = append(findings, Finding{
					Check: "ungitignored-write",
					File:  relPath(root, file),
					Line:  i + 1,
					Message: fmt.Sprintf(
						"open(%q, mode=%q) writes to a path .gitignore does not cover; if this file "+
							"can contain incident content, gitignore it or redact before writing",
						literal, mode),
				})
			}
		}
	}
	return findings, nil
}

func isGitIgnored(root, path string) (bool, error) {
	cmd := exec.Command("git", "-C", root, "check-ignore", "-q", path)
	err := cmd.Run()
	if err == nil {
		return true, nil
	}
	if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
		return false, nil
	}
	return false, err
}

// Scan runs every check against root and returns a stably-sorted finding list.
func Scan(root string) ([]Finding, error) {
	files, err := walkPythonFiles(root)
	if err != nil {
		return nil, err
	}

	var all []Finding
	checks := []func(string, []string) ([]Finding, error){
		checkBypassedGateway,
		checkUnredactedPrint,
		checkUngitignoredWrites,
	}
	for _, check := range checks {
		findings, err := check(root, files)
		if err != nil {
			return nil, err
		}
		all = append(all, findings...)
	}

	sort.Slice(all, func(i, j int) bool {
		if all[i].File != all[j].File {
			return all[i].File < all[j].File
		}
		if all[i].Line != all[j].Line {
			return all[i].Line < all[j].Line
		}
		return all[i].Check < all[j].Check
	})
	return all, nil
}
