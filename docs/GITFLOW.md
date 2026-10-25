# Release workflow

NERT uses GitFlow with `main` as the production branch and `develop` as the
integration branch. Start normal changes on `feature/<name>` from `develop`.

1. Review and validate the feature, then merge it into `develop`.
2. Cut `release/<version>` from the selected integration commit. Freeze the
   feature scope; update the changelog date and release metadata here.
3. Require passing CI and applicable local device checks on the release
   candidate. Carry accepted limitations and deviations into the release notes.
4. Merge the release branch into `main` with a merge commit, and validate that
   exact production commit before creating the annotated `v<version>` tag.
5. Publish that tag and the GitHub release. Never move an existing release tag.
6. Merge `main` back into `develop`, push the back-merge, and verify the tag and
   both branch tips before deleting the finished release branch.

CPU CI runs on feature, integration, release and production branches. GPU
coverage is recorded separately; a disabled or skipped GPU job is not a pass.
Full-catalog AMD validation and persistent AMD CI are accepted known missing
coverage for 0.2.0 and must remain explicit in its published release notes.

Plain Git is sufficient; no GitFlow extension or host setup is required.
