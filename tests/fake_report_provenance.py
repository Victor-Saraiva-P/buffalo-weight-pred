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
