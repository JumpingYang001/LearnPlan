#!/usr/bin/env python3
"""
Safe Markdown Link Fixer
A safe tool that fixes broken markdown links without creating problematic backups.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def extract_number_and_base(filename):
    """Extract number prefix and base name from filename."""
    match = re.match(r'^(\d+)_(.+)\.md$', filename)
    if match:
        return match.group(1), match.group(2)
    return None, filename.replace('.md', '')

def find_best_file_match(expected_filename, search_dir):
    """Find the best matching file using fuzzy matching logic."""
    if not search_dir.exists():
        return None
    
    expected_number, expected_base = extract_number_and_base(expected_filename)
    
    md_files = [f for f in os.listdir(search_dir) if f.endswith('.md') and f != 'README.md']
    
    # First, try to find files with the same number prefix
    if expected_number:
        same_number_files = []
        for f in md_files:
            file_number, _ = extract_number_and_base(f)
            if file_number == expected_number:
                same_number_files.append(f)
        
        if len(same_number_files) == 1:
            return search_dir / same_number_files[0]
    
    # If no unique number match, use similarity matching
    best_match = None
    best_score = 0.0
    threshold = 0.6
    
    for file in md_files:
        score = similarity(expected_filename.lower(), file.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = file
    
    return search_dir / best_match if best_match else None

def should_exclude_from_backup(path):
    """Check if a path should be excluded from backup."""
    path_str = str(path).lower()
    
    # Exclude patterns
    exclude_patterns = [
        '.backup_',
        '__pycache__',
        '.git',
        '.vscode',
        'smartmdtool',
        '.pytest_cache',
        'node_modules',
        '.env'
    ]
    
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    
    # Exclude Python files and other scripts
    if path.suffix.lower() in ['.py', '.pyc', '.pyo']:
        return True
        
    return False

def create_safe_backup(base_path, max_files=100):
    """Create a safe backup with limits and exclusions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = base_path / f".backup_safe_{timestamp}"
    
    print(f"📦 Creating safe backup in {backup_dir.name}...")
    
    file_count = 0
    
    # Only backup .md files from main directories, excluding problematic paths
    for file_path in base_path.rglob("*.md"):
        if should_exclude_from_backup(file_path):
            continue
            
        if file_count >= max_files:
            print(f"⚠️  Backup limited to {max_files} files for safety")
            break
            
        try:
            relative_path = file_path.relative_to(base_path)
            backup_file = backup_dir / relative_path
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_file)
            file_count += 1
        except Exception as e:
            print(f"⚠️  Skipped {file_path}: {e}")
    
    print(f"✅ Safe backup created: {backup_dir.name} ({file_count} files)")
    return backup_dir

def process_file(file_path, dry_run=True):
    """Process a single markdown file and fix broken links."""
    fixes_made = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"    ❌ Error reading {file_path}: {e}")
        return 0
    
    original_content = content
    
    # Pattern to match "See details in filename.md" links
    pattern = r'\[See details in ([^]]+\.md)\]\(([^)]+)\)'
    
    def fix_link(match):
        nonlocal fixes_made
        expected_file = match.group(1)
        current_link = match.group(2)
        
        print(f"    🔍 Found link: '{expected_file}' -> '{current_link}'")
        
        # Check if the current link target actually exists
        full_target_path = file_path.parent / current_link
        
        if full_target_path.exists():
            # Link works, check if text matches filename
            actual_filename = full_target_path.name
            if expected_file != actual_filename:
                fixes_made += 1
                print(f"    🔧 {'Would fix' if dry_run else 'Fixed'} link text: '{expected_file}' -> '{actual_filename}'")
                if not dry_run:
                    return f'[See details in {actual_filename}]({current_link})'
        else:
            print(f"    ❌ Link target missing: {full_target_path}")
            # Link is broken, try to find the correct file
            link_path = Path(current_link)
            target_dir = file_path.parent / link_path.parent
            
            # Try to find the correct file
            correct_file = find_best_file_match(expected_file, target_dir)
            
            if correct_file and correct_file.exists():
                try:
                    relative_path = correct_file.relative_to(file_path.parent)
                    new_link = str(relative_path).replace('\\', '/')
                    fixes_made += 1
                    print(f"    🔧 {'Would fix' if dry_run else 'Fixed'} broken link:")
                    print(f"        From: '{current_link}' -> '{new_link}'")
                    print(f"        Text: '{expected_file}' -> '{correct_file.name}'")
                    if not dry_run:
                        return f'[See details in {correct_file.name}]({new_link})'
                except Exception as e:
                    print(f"    ❌ Could not create relative path: {e}")
            else:
                print(f"    ❌ No match found for '{expected_file}'")
        
        return match.group(0)  # No change needed
    
    # Apply fixes
    if not dry_run:
        content = re.sub(pattern, fix_link, content)
        
        # Also handle "See project details" links
        project_pattern = r'\[See project details in ([^]]+\.md)\]\(([^)]+)\)'
        content = re.sub(project_pattern, fix_link, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"    ✅ Updated {file_path.name}")
    else:
        # Just scan for issues
        re.sub(pattern, fix_link, content)
        project_pattern = r'\[See project details in ([^]]+\.md)\]\(([^)]+)\)'
        re.sub(project_pattern, fix_link, content)
    
    return fixes_made

def main():
    """Main function to fix broken links"""
    # Always run from the parent directory of SmartMDTool
    script_dir = Path(__file__).parent
    base_path = script_dir.parent
    
    print("🛡️  Safe Markdown Link Fixer")
    print("=" * 50)
    print(f"Working directory: {base_path}")
    
    # Ask user for confirmation
    print("\n🔍 This tool will:")
    print("  1. Create a safe backup (limited to .md files, excluding problematic directories)")
    print("  2. Scan for broken markdown links")
    print("  3. Fix mismatched link text and broken links")
    
    response = input("\n❓ Do you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Operation cancelled by user")
        return
    
    # Create safe backup
    backup_dir = create_safe_backup(base_path)
    
    total_fixes = 0
    files_processed = 0
    
    # Process specific main topic files
    topic_dirs = [
        '01_C_CPP_Core_Programming', '02_Networking_Protocols', 
        '03_Debugging_Performance', '04_Systems_Programming',
        '05_Build_Systems_Tools', '06_Testing_Frameworks',
        '07_Graphics_UI_Programming', '08_Machine_Learning_AI',
        '09_Parallel_Computing', '10_Robotics_Automotive',
        '11_Cloud_Distributed_Systems', '12_Industry_Protocols'
    ]
    
    # First do a dry run
    print("\n🔍 DRY RUN - Scanning for issues...")
    for topic_dir in topic_dirs:
        topic_path = base_path / topic_dir
        if not topic_path.exists():
            continue
            
        print(f"\n📁 Scanning {topic_dir}")
        
        # Process all .md files in the topic directory (main topic files)
        for md_file in topic_path.glob("*.md"):
            print(f"  🔍 Checking {md_file.name}...")
            fixes = process_file(md_file, dry_run=True)
            total_fixes += fixes
            files_processed += 1
    
    if total_fixes == 0:
        print("\n✅ No issues found!")
        return
    
    print(f"\n📊 Found {total_fixes} issues in {files_processed} files")
    
    # Ask for confirmation to apply fixes
    response = input("❓ Apply fixes? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Fixes not applied")
        return
    
    # Apply fixes
    print("\n🔧 Applying fixes...")
    total_fixes = 0
    files_processed = 0
    
    for topic_dir in topic_dirs:
        topic_path = base_path / topic_dir
        if not topic_path.exists():
            continue
            
        print(f"\n📁 Fixing {topic_dir}")
        
        for md_file in topic_path.glob("*.md"):
            print(f"  🔧 Fixing {md_file.name}...")
            fixes = process_file(md_file, dry_run=False)
            total_fixes += fixes
            files_processed += 1
    
    print(f"\n🎉 Summary:")
    print(f"Files processed: {files_processed}")
    print(f"Total fixes made: {total_fixes}")
    print(f"Backup location: {backup_dir.name}")

if __name__ == "__main__":
    main()
