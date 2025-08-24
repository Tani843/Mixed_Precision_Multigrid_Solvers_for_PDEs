#!/usr/bin/env python3
"""
Automated Release Management Script

This script handles version bumping, changelog generation, and release preparation
for the mixed-precision multigrid package.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import toml


class ReleaseManager:
    """Manages package releases with automated version bumping and changelog generation."""
    
    def __init__(self, repo_root: Optional[str] = None):
        """Initialize the release manager."""
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.version_file = self.repo_root / "src" / "multigrid" / "_version.py"
        self.pyproject_file = self.repo_root / "pyproject.toml"
        self.changelog_file = self.repo_root / "CHANGELOG.md"
        
        # Version pattern for semantic versioning
        self.version_pattern = re.compile(r'__version__ = ["\'](\d+\.\d+\.\d+(?:-\w+)?)["\']')
        
    def get_current_version(self) -> str:
        """Get the current version from _version.py."""
        if not self.version_file.exists():
            raise FileNotFoundError(f"Version file not found: {self.version_file}")
        
        content = self.version_file.read_text()
        match = self.version_pattern.search(content)
        if not match:
            raise ValueError("Could not parse version from _version.py")
        
        return match.group(1)
    
    def bump_version(self, version_type: str, prerelease: Optional[str] = None) -> str:
        """
        Bump version according to semantic versioning.
        
        Args:
            version_type: 'major', 'minor', or 'patch'
            prerelease: Optional prerelease identifier (alpha, beta, rc)
        
        Returns:
            New version string
        """
        current = self.get_current_version()
        
        # Parse current version
        if '-' in current:
            base_version, _ = current.split('-', 1)
        else:
            base_version = current
        
        major, minor, patch = map(int, base_version.split('.'))
        
        # Bump version
        if version_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif version_type == 'minor':
            minor += 1
            patch = 0
        elif version_type == 'patch':
            patch += 1
        else:
            raise ValueError(f"Invalid version type: {version_type}")
        
        new_version = f"{major}.{minor}.{patch}"
        
        # Add prerelease suffix if specified
        if prerelease:
            new_version += f"-{prerelease}"
        
        return new_version
    
    def update_version_file(self, new_version: str) -> None:
        """Update the version file with new version and build info."""
        content = self.version_file.read_text()
        
        # Update version
        content = self.version_pattern.sub(f'__version__ = "{new_version}"', content)
        
        # Update build metadata
        build_date = datetime.datetime.now().isoformat()
        build_hash = self.get_git_hash()
        build_branch = self.get_git_branch()
        
        # Replace build metadata
        content = re.sub(
            r'BUILD_DATE = .*',
            f'BUILD_DATE = "{build_date}"',
            content
        )
        content = re.sub(
            r'BUILD_HASH = .*',
            f'BUILD_HASH = "{build_hash}"',
            content
        )
        content = re.sub(
            r'BUILD_BRANCH = .*',
            f'BUILD_BRANCH = "{build_branch}"',
            content
        )
        
        self.version_file.write_text(content)
        print(f"✓ Updated version to {new_version}")
    
    def get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return result.stdout.strip()[:8]  # Short hash
        except subprocess.SubprocessError:
            return "unknown"
    
    def get_git_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return result.stdout.strip()
        except subprocess.SubprocessError:
            return "unknown"
    
    def get_commits_since_tag(self, tag: Optional[str] = None) -> List[str]:
        """Get commit messages since last tag."""
        try:
            if tag:
                cmd = ["git", "log", f"{tag}..HEAD", "--oneline"]
            else:
                # Get all commits if no tag specified
                cmd = ["git", "log", "--oneline", "-20"]  # Last 20 commits
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except subprocess.SubprocessError:
            return []
    
    def categorize_commits(self, commits: List[str]) -> Dict[str, List[str]]:
        """Categorize commits by type (feat, fix, docs, etc.)."""
        categories = {
            'Features': [],
            'Bug Fixes': [],
            'Documentation': [],
            'Performance': [],
            'Refactoring': [],
            'Tests': [],
            'Other': []
        }
        
        for commit in commits:
            if not commit.strip():
                continue
            
            # Extract commit message (after hash)
            parts = commit.split(' ', 1)
            if len(parts) < 2:
                continue
            
            message = parts[1]
            
            # Categorize based on conventional commits or keywords
            if any(keyword in message.lower() for keyword in ['feat:', 'feature:', 'add:', 'implement:']):
                categories['Features'].append(message)
            elif any(keyword in message.lower() for keyword in ['fix:', 'bug:', 'resolve:', 'patch:']):
                categories['Bug Fixes'].append(message)
            elif any(keyword in message.lower() for keyword in ['docs:', 'doc:', 'documentation:']):
                categories['Documentation'].append(message)
            elif any(keyword in message.lower() for keyword in ['perf:', 'performance:', 'optimize:', 'speed:']):
                categories['Performance'].append(message)
            elif any(keyword in message.lower() for keyword in ['refactor:', 'restructure:', 'cleanup:']):
                categories['Refactoring'].append(message)
            elif any(keyword in message.lower() for keyword in ['test:', 'tests:', 'testing:']):
                categories['Tests'].append(message)
            else:
                categories['Other'].append(message)
        
        return categories
    
    def generate_changelog_entry(self, version: str, commits: List[str]) -> str:
        """Generate changelog entry for new version."""
        date = datetime.date.today().isoformat()
        
        entry = f"\n## [{version}] - {date}\n\n"
        
        if not commits:
            entry += "- Initial release\n"
            return entry
        
        categories = self.categorize_commits(commits)
        
        for category, commit_list in categories.items():
            if commit_list:
                entry += f"### {category}\n\n"
                for commit in commit_list:
                    # Clean up commit message
                    clean_commit = re.sub(r'^(feat|fix|docs|perf|refactor|test):\s*', '', commit, flags=re.IGNORECASE)
                    entry += f"- {clean_commit}\n"
                entry += "\n"
        
        return entry
    
    def update_changelog(self, version: str) -> None:
        """Update changelog with new version entry."""
        last_tag = self.get_last_git_tag()
        commits = self.get_commits_since_tag(last_tag)
        
        new_entry = self.generate_changelog_entry(version, commits)
        
        if self.changelog_file.exists():
            content = self.changelog_file.read_text()
            # Insert new entry after header
            lines = content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('##') and '[' in line:
                    header_end = i
                    break
            
            if header_end > 0:
                # Insert before first existing version
                lines.insert(header_end, new_entry.rstrip())
            else:
                # Append to end if no existing versions found
                content += new_entry
                lines = content.split('\n')
            
            self.changelog_file.write_text('\n'.join(lines))
        else:
            # Create new changelog
            header = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n"
            self.changelog_file.write_text(header + new_entry)
        
        print(f"✓ Updated changelog with {len(commits)} commits")
    
    def get_last_git_tag(self) -> Optional[str]:
        """Get the last git tag."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except subprocess.SubprocessError:
            return None
    
    def create_git_tag(self, version: str) -> None:
        """Create a git tag for the new version."""
        tag_name = f"v{version}"
        try:
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
                cwd=self.repo_root,
                check=True
            )
            print(f"✓ Created git tag {tag_name}")
        except subprocess.SubprocessError as e:
            print(f"⚠ Failed to create git tag: {e}")
    
    def commit_changes(self, version: str) -> None:
        """Commit version changes."""
        try:
            # Add changed files
            subprocess.run(
                ["git", "add", str(self.version_file), str(self.changelog_file)],
                cwd=self.repo_root,
                check=True
            )
            
            # Commit with version message
            subprocess.run(
                ["git", "commit", "-m", f"Release version {version}"],
                cwd=self.repo_root,
                check=True
            )
            print(f"✓ Committed version {version}")
        except subprocess.SubprocessError as e:
            print(f"⚠ Failed to commit changes: {e}")
    
    def validate_working_directory(self) -> bool:
        """Check if working directory is clean."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return len(result.stdout.strip()) == 0
        except subprocess.SubprocessError:
            return False
    
    def prepare_release(self, version_type: str, prerelease: Optional[str] = None,
                       commit: bool = True, tag: bool = True) -> str:
        """
        Prepare a new release.
        
        Args:
            version_type: 'major', 'minor', or 'patch'
            prerelease: Optional prerelease identifier
            commit: Whether to commit changes
            tag: Whether to create git tag
        
        Returns:
            New version string
        """
        print("🚀 Preparing new release...")
        
        # Validate preconditions
        if commit and not self.validate_working_directory():
            print("❌ Working directory is not clean. Commit or stash changes first.")
            sys.exit(1)
        
        # Get new version
        new_version = self.bump_version(version_type, prerelease)
        current_version = self.get_current_version()
        
        print(f"📦 Version: {current_version} → {new_version}")
        
        # Update files
        self.update_version_file(new_version)
        self.update_changelog(new_version)
        
        # Git operations
        if commit:
            self.commit_changes(new_version)
        
        if tag:
            self.create_git_tag(new_version)
        
        print(f"✅ Release {new_version} prepared successfully!")
        
        return new_version


def main():
    """Main entry point for release management script."""
    parser = argparse.ArgumentParser(description="Automated release management")
    parser.add_argument(
        "version_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--prerelease",
        help="Prerelease identifier (alpha, beta, rc)"
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Don't commit changes"
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Don't create git tag"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    try:
        manager = ReleaseManager()
        
        if args.dry_run:
            current = manager.get_current_version()
            new_version = manager.bump_version(args.version_type, args.prerelease)
            print(f"DRY RUN: Would bump version {current} → {new_version}")
            return
        
        new_version = manager.prepare_release(
            args.version_type,
            prerelease=args.prerelease,
            commit=not args.no_commit,
            tag=not args.no_tag
        )
        
        print(f"\n📋 Next steps:")
        print(f"   1. Review changes: git show HEAD")
        print(f"   2. Push changes: git push origin main")
        print(f"   3. Push tags: git push origin v{new_version}")
        print(f"   4. Create GitHub release from tag")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()