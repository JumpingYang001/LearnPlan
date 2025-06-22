#!/usr/bin/env python3
"""
Script to systematically verify and update all main topic .md files
to ensure every section and project contains correct links to corresponding
subtopic or project .md files in their subfolders.

Usage:
    python update_links.py           # Process only [Start] files and mark as [Done]
    python update_links.py --cleanup # Clean up duplicate links in all [Done] files

Features:
    - Automatically detects sections (### headers) and projects (numbered items)
    - Generates standardized links for both sections and projects
    - Removes duplicate and outdated links
    - Updates topic.txt status tracking
    - Handles various link formats and edge cases
"""

import os
import re
import sys
from pathlib import Path

def read_topic_file(topic_file_path):
    """Read the topic.txt file and return list of files to process."""
    files_to_process = []
    all_files = []
    try:
        with open(topic_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('[Start]'):
                    # Extract the file path
                    file_path = line.replace('[Start]', '').strip()
                    files_to_process.append(file_path)
                elif line.startswith('[Done]'):
                    # Also include done files for duplicate cleanup
                    file_path = line.replace('[Done]', '').strip()
                    all_files.append(file_path)
                    
        # Return both lists
        return files_to_process, all_files
    except FileNotFoundError:
        print(f"Error: {topic_file_path} not found")
        sys.exit(1)
    return files_to_process, []

def read_md_file(file_path):
    """Read markdown file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            print(f"Error: Could not read {file_path}")
            return None

def write_md_file(file_path, content):
    """Write content to markdown file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return False

def extract_folder_name(file_path):
    """Extract the folder name from file path for link construction."""
    # Convert path like "07_Graphics_UI_Programming/01_Qt_QML.md" 
    # to folder name like "01_Qt_QML"
    parts = file_path.split('/')
    if len(parts) >= 2:
        filename = parts[-1]  # e.g., "01_Qt_QML.md"
        return filename.replace('.md', '')  # e.g., "01_Qt_QML"
    return None

def find_sections_and_projects(content):
    """Find all sections (###) and projects in the markdown content."""
    sections = []
    projects = []
    
    lines = content.split('\n')
    in_projects = False
    
    for i, line in enumerate(lines):
        # Check for section headers
        if line.startswith('### '):
            section_match = re.match(r'### (\d+)\.\s*(.+?)\s*\(', line)
            if section_match:
                section_num = section_match.group(1)
                section_title = section_match.group(2)
                sections.append({
                    'number': section_num,
                    'title': section_title,
                    'line_index': i,
                    'full_line': line
                })
        
        # Check for projects section
        if line.strip() == '## Projects':
            in_projects = True
            continue
        
        # Check for end of projects section
        if in_projects and line.startswith('## ') and line.strip() != '## Projects':
            in_projects = False
        
        # Find numbered projects
        if in_projects and re.match(r'^\d+\.\s*\*\*', line):
            project_match = re.match(r'^(\d+)\.\s*\*\*(.+?)\*\*', line)
            if project_match:
                project_num = project_match.group(1)
                project_title = project_match.group(2)
                projects.append({
                    'number': project_num,
                    'title': project_title,
                    'line_index': i,
                    'full_line': line
                })
    
    return sections, projects

def has_link_in_next_line(lines, line_index):
    """Check if the next line contains a [See details in ...] link."""
    if line_index + 1 < len(lines):
        next_line = lines[line_index + 1].strip()
        return '[See details in' in next_line or '[See project details' in next_line
    return False

def find_existing_links_after_line(lines, line_index, max_search_lines=10):
    """Find any existing links after a given line index."""
    existing_links = []
    search_end = min(line_index + max_search_lines + 1, len(lines))
    
    for i in range(line_index + 1, search_end):
        line = lines[i].strip()
        # Check for any markdown link format
        if re.search(r'\[.*?\]\(.*?\)', line):
            # Check if it's a details/project link
            if any(keyword in line for keyword in ['See details', 'See project details', 'project_', 'Projects/']):
                existing_links.append(i)
        # Stop searching if we hit another section/project header or major section
        elif line.startswith('###') or line.startswith('##') or re.match(r'^\d+\.\s*\*\*', line):
            break
    
    return existing_links

def clean_existing_links(lines, line_index, max_search_lines=10):
    """Remove existing links after a given line index."""
    existing_link_indices = find_existing_links_after_line(lines, line_index, max_search_lines)
    
    # Remove lines in reverse order to maintain indices
    for link_index in reversed(existing_link_indices):
        lines.pop(link_index)
    
    return len(existing_link_indices)

def generate_section_link(section_num, section_title, folder_name):
    """Generate a section link."""
    # Convert section title to filename format
    # Remove special characters and replace spaces with underscores
    clean_title = re.sub(r'[^\w\s-]', '', section_title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    
    filename = f"{section_num:02d}_{clean_title}.md"
    link = f"[See details in {filename}]({folder_name}/{filename})"
    return link

def generate_project_link(project_num, project_title, folder_name):
    """Generate a project link."""
    # Convert project title to filename format
    clean_title = re.sub(r'[^\w\s-]', '', project_title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    
    filename = f"project_{project_num:02d}_{clean_title}.md"
    link = f"[See project details in {filename}]({folder_name}/{filename})"
    return link

def update_md_file_links(content, sections, projects, folder_name):
    """Update the markdown content with missing links."""
    lines = content.split('\n')
    modified = False
    
    # Process sections
    for section in reversed(sections):  # Process in reverse to maintain line indices
        line_index = section['line_index']
        
        # Clean any existing links first
        removed_count = clean_existing_links(lines, line_index)
        if removed_count > 0:
            modified = True
            # Update line indices for all items after this section
            for other_section in sections:
                if other_section['line_index'] > line_index:
                    other_section['line_index'] -= removed_count
            for project in projects:
                if project['line_index'] > line_index:
                    project['line_index'] -= removed_count
        
        # Add the new standardized link
        link = generate_section_link(int(section['number']), section['title'], folder_name)
        lines.insert(line_index + 1, link)
        modified = True
        
        # Update line indices for subsequent items
        for other_section in sections:
            if other_section['line_index'] > line_index:
                other_section['line_index'] += 1
        for project in projects:
            if project['line_index'] > line_index:
                project['line_index'] += 1
    
    # Process projects
    for project in reversed(projects):  # Process in reverse to maintain line indices
        line_index = project['line_index']
        
        # Clean any existing links first
        removed_count = clean_existing_links(lines, line_index)
        if removed_count > 0:
            modified = True
            # Update line indices for all items after this project
            for other_project in projects:
                if other_project['line_index'] > line_index:
                    other_project['line_index'] -= removed_count
        
        # Add the new standardized link
        link = generate_project_link(int(project['number']), project['title'], folder_name)
        lines.insert(line_index + 1, f"   {link}")
        modified = True
        
        # Update line indices for subsequent projects
        for other_project in projects:
            if other_project['line_index'] > line_index:
                other_project['line_index'] += 1
    
    return '\n'.join(lines), modified

def update_topic_status(topic_file_path, file_path, status='Done'):
    """Update the status of a file in topic.txt."""
    try:
        with open(topic_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace [Start] with [Done] for the specific file
        old_line = f"[Start] {file_path}"
        new_line = f"[{status}] {file_path}"
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            with open(topic_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"Warning: Could not find '{old_line}' in topic.txt")
            return False
    except Exception as e:
        print(f"Error updating topic.txt: {e}")
        return False

def process_file(base_path, file_path, topic_file_path, cleanup_mode=False):
    """Process a single markdown file."""
    full_path = os.path.join(base_path, file_path)
    print(f"Processing: {file_path}")
    
    # Read the file
    content = read_md_file(full_path)
    if content is None:
        return False
    
    # Extract folder name for links
    folder_name = extract_folder_name(file_path)
    if folder_name is None:
        print(f"Error: Could not extract folder name from {file_path}")
        return False
    
    # Find sections and projects
    sections, projects = find_sections_and_projects(content)
    
    print(f"  Found {len(sections)} sections and {len(projects)} projects")
    
    # Update content with missing links
    updated_content, modified = update_md_file_links(content, sections, projects, folder_name)
    
    if modified:
        # Write the updated content
        if write_md_file(full_path, updated_content):
            if cleanup_mode:
                print(f"  Cleaned up duplicate links in {file_path}")
            else:
                print(f"  Updated {file_path} with missing links")
            # Update topic.txt status (only if not in cleanup mode)
            if not cleanup_mode:
                update_topic_status(topic_file_path, file_path)
            return True
        else:
            return False
    else:
        if cleanup_mode:
            print(f"  No duplicate links found in {file_path}")
        else:
            print(f"  No links needed for {file_path}")
            # Still mark as done since it's verified (only if not in cleanup mode)
            if not cleanup_mode:
                update_topic_status(topic_file_path, file_path)
        return True

def main():
    # Set up paths
    base_path = Path(__file__).parent
    topic_file_path = base_path / "topic.txt"
    
    print("Starting systematic link verification and update...")
    print(f"Base path: {base_path}")
    print(f"Topic file: {topic_file_path}")
    
    # Check command line arguments for cleanup mode
    cleanup_mode = len(sys.argv) > 1 and sys.argv[1] == '--cleanup'
    
    # Read files to process
    files_to_process, all_done_files = read_topic_file(topic_file_path)
    
    if cleanup_mode:
        print("\n=== CLEANUP MODE: Processing all files to remove duplicates ===")
        files_to_process = all_done_files
        print(f"Found {len(files_to_process)} done files to clean up")
    else:
        print(f"Found {len(files_to_process)} files to process")
        if all_done_files:
            print(f"Also found {len(all_done_files)} done files (use --cleanup to process these)")
    
    # Process each file
    processed_count = 0
    failed_count = 0
    
    for file_path in files_to_process:
        try:
            if process_file(base_path, file_path, topic_file_path, cleanup_mode):
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(files_to_process)}")
    
    if not cleanup_mode and all_done_files:
        print(f"\nTo clean up duplicate links in done files, run:")
        print(f"python {sys.argv[0]} --cleanup")

if __name__ == "__main__":
    main()
