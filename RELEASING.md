# Releasing

1. Make sure `master` is in the state you want to release.

2. Create a GitHub release with the new version tag (e.g. `v0.13.0`):

   ```bash
   gh release create v0.13.0 --repo sign-language-processing/pose --title "v0.13.0" --notes "..."
   ```

   Or do it in the GitHub UI: **Releases → Draft a new release → Choose a tag → Create new tag**.

3. That's it. The [`pypi-publish` workflow](.github/workflows/pypi-publish.yaml) fires automatically on release creation. It:
   - reads the tag name as the version,
   - patches `pyproject.toml` and commits it back to `master`, and
   - builds and publishes to PyPI.

There is no need to manually edit `pyproject.toml` before the release — for example, the version field does not need to be changed manually.
