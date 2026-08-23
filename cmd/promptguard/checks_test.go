package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func newTempRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	run("config", "user.email", "test@example.com")
	run("config", "user.name", "test")
	return dir
}

func TestCheckBypassedGateway_FlagsDirectConstruction(t *testing.T) {
	root := newTempRepo(t)
	writeFile(t, filepath.Join(root, "azure_auth.py"), "client = AzureOpenAI(api_key=key)\n")
	writeFile(t, filepath.Join(root, "scratch.py"), "from openai import AzureOpenAI\nclient = AzureOpenAI(api_key=key)\n")

	findings, err := checkBypassedGateway(root, []string{
		filepath.Join(root, "azure_auth.py"),
		filepath.Join(root, "scratch.py"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].File != "scratch.py" {
		t.Errorf("expected finding in scratch.py, got %s", findings[0].File)
	}
	if findings[0].Line != 2 {
		t.Errorf("expected line 2, got %d", findings[0].Line)
	}
}

func TestCheckBypassedGateway_SanctionedFactoryClean(t *testing.T) {
	root := newTempRepo(t)
	writeFile(t, filepath.Join(root, "azure_auth.py"), "client = AzureOpenAI(api_key=key)\n")

	findings, err := checkBypassedGateway(root, []string{filepath.Join(root, "azure_auth.py")})
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected no findings in the sanctioned factory, got %+v", findings)
	}
}

func TestCheckUnredactedPrint_FlagsSensitiveInterpolation(t *testing.T) {
	root := newTempRepo(t)
	writeFile(t, filepath.Join(root, "debug.py"),
		"print(f\"System prompt: {system_prompt}\")\n"+
			"print(\"totally fine\")\n"+
			"print(f\"redacted: {guard.redact_text(user_prompt)}\")\n",
	)

	findings, err := checkUnredactedPrint(root, []string{filepath.Join(root, "debug.py")})
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Line != 1 {
		t.Errorf("expected line 1, got %d", findings[0].Line)
	}
}

func TestCheckUngitignoredWrites(t *testing.T) {
	root := newTempRepo(t)
	writeFile(t, filepath.Join(root, ".gitignore"), "logs/\nerror.log\n")
	writeFile(t, filepath.Join(root, "app.py"),
		"with open('logs/debug.log', 'w') as f:\n"+
			"    pass\n"+
			"with open('leak.log', 'w') as f:\n"+
			"    pass\n"+
			"with open(f'logs/fetcher_debug_{incident_number}.log', 'w') as f:\n"+
			"    pass\n",
	)

	findings, err := checkUngitignoredWrites(root, []string{filepath.Join(root, "app.py")})
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding (leak.log), got %d: %+v", len(findings), findings)
	}
	if findings[0].Line != 3 {
		t.Errorf("expected line 3 (leak.log), got %d: %+v", findings[0].Line, findings)
	}
}

func TestScan_CleanRepoHasNoFindings(t *testing.T) {
	root := newTempRepo(t)
	writeFile(t, filepath.Join(root, ".gitignore"), "logs/\n")
	writeFile(t, filepath.Join(root, "azure_auth.py"), "client = AzureOpenAI(api_key=key)\n")
	writeFile(t, filepath.Join(root, "app.py"),
		"print('hello world')\n"+
			"with open('logs/app.log', 'w') as f:\n"+
			"    pass\n",
	)

	findings, err := Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected a clean scan, got %+v", findings)
	}
}
