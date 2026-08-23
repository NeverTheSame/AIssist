// Command promptguard is a static checker for the AIssist repo: it flags
// AzureOpenAI clients built outside the one sanctioned factory (a bypass
// of guard.wrap_client), print() calls that leak sensitive variables past
// the log-redaction filter, and file writes to paths .gitignore doesn't
// cover. Wired into CI (.github/workflows/security.yml) and
// .pre-commit-config.yaml.
//
// Exit codes: 0 clean, 1 findings, 2 tool error.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

func main() {
	root := flag.String("root", ".", "repository root to scan")
	format := flag.String("format", "text", `output format: "text" or "json"`)
	flag.Parse()

	findings, err := Scan(*root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "promptguard: %v\n", err)
		os.Exit(2)
	}

	if findings == nil {
		findings = []Finding{}
	}

	switch *format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(findings); err != nil {
			fmt.Fprintf(os.Stderr, "promptguard: %v\n", err)
			os.Exit(2)
		}
	case "text":
		if len(findings) == 0 {
			fmt.Println("promptguard: clean -- no findings")
		} else {
			for _, f := range findings {
				fmt.Printf("%s:%d: [%s] %s\n", f.File, f.Line, f.Check, f.Message)
			}
			fmt.Fprintf(os.Stderr, "\npromptguard: %d finding(s)\n", len(findings))
		}
	default:
		fmt.Fprintf(os.Stderr, "promptguard: unknown --format %q (want text or json)\n", *format)
		os.Exit(2)
	}

	if len(findings) > 0 {
		os.Exit(1)
	}
}
