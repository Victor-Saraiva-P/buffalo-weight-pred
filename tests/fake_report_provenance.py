from __future__ import annotations


class FixedReportProvenance:
    def inputs_recipe_hash(self) -> str:
        """Return fixed knowledge.

        Example: this isolates recipe discovery in a test.
        """
        return "1" * 64

    def dependencies(self) -> dict[str, str]:
        """Return fixed packages.

        Example: this isolates environment discovery.
        """
        return {"fake-compute": "1.0"}

    def repository_commit(self) -> str:
        """Return fixed source identity.

        Example: this isolates the Git process.
        """
        return "2" * 40


class FixedFeatureSelectionProvenance:
    def feature_selection_recipe_hash(self) -> str:
        """Return fixed selection knowledge; for example, tests get stable manifests."""
        return "3" * 64

    def feature_selection_dependencies(self) -> dict[str, str]:
        """Return fixed selection packages; for example, tests avoid environment coupling."""
        return {"fake-selection": "2.0"}

    def repository_commit(self) -> str:
        """Return fixed source identity; for example, tests avoid invoking Git."""
        return "2" * 40
