# Preparing a GitHub release

This document describes manual publication. No script in this repository pushes
code, creates a GitHub repository, or publishes a release.

## Source and attribution

Preserve `LICENSE`, `README.legommenders.md`, and the upstream attribution in the
root README. Keep the upstream Git history when maintaining a fork. Set your own
repository URL in the README clone example before publication.

A prepared copy can keep `upstream` pointing at Jyonn/Legommenders and have no
`origin` until you create or choose your own repository. Do not push to upstream
by accident. Review the working tree and create your release commit locally
before you decide to publish.

## Checks

1. Install using the root README in a new checkout and a new Python environment.
   Leave `.data` absent. Run doctor, backend tests, frontend tests, and build.
2. Open the built application, load a template, validate/compile/export YAML,
   and save/reload a graph without preinstalled datasets.
3. Run a small real CPU or GPU training and evaluation workflow with automatic
   acquisition. Check a second run reuses compatible caches. Verify local `.data`
   overrides independently. Record actual results and any provider failures.
4. Review `git status --short` and `git diff --check`. Confirm `.data`, `.auth`,
   real `.env` files, TLS private keys, environments, `data/`, `cache/`,
   `checkpoints/`, `node_modules/`, `dist/`, logs, and workspace are not tracked.
   `.gitignore` does not remove files that were committed previously.
5. Scan the staged source and history for credentials, private URLs, unintended
   machine paths, and large blobs. Test fixture tokens are intentionally fake;
   review matches instead of assuming every string is a real credential.
6. Keep dataset/model assets outside Git. Provide acquisition instructions and
   respect their publishers' access and redistribution conditions. Checkpoint
   distribution should be deliberate and separate from the source repository.
7. Review CI after you later push. The workflow is only a configuration until
   GitHub actually runs it; local test results are not a GitHub Actions result.

An initial source archive can be made from the reviewed Git commit using
`git archive`. Do not archive the whole working directory after running tests or
training, because ignored environments/caches can be many GB. GitHub publication,
repository visibility, tags, and release attachments are separate manual steps.

## Suggested source release contents

- Complete visual application and compatible Legommenders core.
- English root README, data/development guides, optional `.data.example`.
- Installation, diagnosis, launch, and scoped stop scripts.
- Python dependency constraints and frontend package-lock.
- Tests and CPU GitHub Actions workflow.
- Validation notes stating what ran, on which platform, and any unverified paths.
