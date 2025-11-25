#!/usr/bin/env python3
"""
Generate markdown documentation from proto-control Rust source.
"""

import re
import subprocess
from pathlib import Path


def extract_docs_and_code(rust_file):
    """Extract documentation comments and code from Rust source."""
    with open(rust_file, 'r') as f:
        lines = f.readlines()

    # Extract module-level docs
    module_doc = []
    i = 0
    while i < len(lines) and lines[i].startswith('//!'):
        module_doc.append(lines[i][3:].strip())
        i += 1

    module_doc_text = '\n'.join(module_doc)

    # Parse types and their docs
    items = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for doc comment start
        if line.strip().startswith('///') and not line.strip().startswith('////'):
            # Collect all consecutive doc comments
            doc_lines = []
            j = i
            while j < len(lines) and lines[j].strip().startswith('///'):
                # Preserve indentation after the ///
                doc_line = lines[j].strip()[3:]
                if doc_line.startswith(' '):
                    doc_line = doc_line[1:]  # Remove single leading space after ///
                doc_lines.append(doc_line.rstrip())
                j += 1

            # Skip blank lines and look for derives or pub struct/trait
            while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('#[')):
                j += 1

            # Check if next non-empty, non-derive line is pub struct/trait
            if j < len(lines) and re.match(r'^\s*pub\s+(struct|trait)\s+', lines[j]):
                # Capture derives and definition
                code_start = j
                # Back up to capture derives
                k = j - 1
                while k >= i and (lines[k].strip() == '' or lines[k].strip().startswith('#[')):
                    if lines[k].strip().startswith('#['):
                        code_start = k
                    k -= 1

                # Extract type info
                type_match = re.search(r'(struct|trait)\s+(\w+)', lines[j])
                if type_match:
                    type_kind = type_match.group(1)
                    type_name = type_match.group(2)

                    # Capture the full definition (including body for struct/trait)
                    code_lines = []
                    brace_count = 0
                    is_tuple_struct = '(' in lines[j] and ');' in lines[j]

                    m = code_start
                    while m < len(lines):
                        code_lines.append(lines[m].rstrip())

                        if is_tuple_struct and lines[m].strip().endswith(');'):
                            m += 1
                            break

                        brace_count += lines[m].count('{') - lines[m].count('}')
                        m += 1

                        if brace_count == 0 and '{' in ''.join(code_lines):
                            break

                    items.append({
                        'kind': type_kind,
                        'name': type_name,
                        'doc': '\n'.join(doc_lines),
                        'code': '\n'.join(code_lines)
                    })

                i = m
            else:
                i += 1
        else:
            i += 1

    # Extract impl blocks with documented methods
    content = ''.join(lines)
    impl_pattern = r'impl\s+(\w+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}'
    impls = {}

    for match in re.finditer(impl_pattern, content):
        type_name = match.group(1)
        impl_body = match.group(2)

        # Extract public methods/consts with docs
        methods = []
        method_lines = impl_body.split('\n')

        j = 0
        while j < len(method_lines):
            if method_lines[j].strip().startswith('///'):
                # Collect doc comments
                method_doc_lines = []
                k = j
                while k < len(method_lines) and method_lines[k].strip().startswith('///'):
                    # Preserve indentation after the ///
                    doc_line = method_lines[k].strip()[3:]
                    if doc_line.startswith(' '):
                        doc_line = doc_line[1:]  # Remove single leading space after ///
                    method_doc_lines.append(doc_line.rstrip())
                    k += 1

                # Look for pub fn or pub const
                while k < len(method_lines) and method_lines[k].strip() == '':
                    k += 1

                if k < len(method_lines) and re.match(r'^\s*pub\s+(fn|const)\s+', method_lines[k]):
                    sig_match = re.search(r'pub\s+(?:fn|const)\s+(\w+)', method_lines[k])
                    if sig_match:
                        method_name = sig_match.group(1)

                        # Extract full signature (up to { or ;)
                        sig_lines = []
                        m = k
                        while m < len(method_lines):
                            line = method_lines[m]
                            # Stop at opening brace or semicolon
                            if '{' in line:
                                # Add everything up to the brace
                                sig_lines.append(line.split('{')[0].strip())
                                break
                            elif ';' in line:
                                sig_lines.append(line.split(';')[0].strip())
                                break
                            else:
                                sig_lines.append(line.strip())
                            m += 1

                        methods.append({
                            'name': method_name,
                            'doc': '\n'.join(method_doc_lines),
                            'signature': ' '.join(sig_lines)
                        })
                        j = m
                else:
                    j = k
            else:
                j += 1

        if methods:
            impls[type_name] = methods

    return module_doc_text, items, impls


def generate_markdown(module_doc, items, impls):
    """Generate markdown documentation."""
    md = []

    # Title
    md.append("# Proto-Control API Specification\n\n")
    md.append("**External Integrator Interface Documentation**\n\n")

    # Module-level documentation
    if module_doc:
        md.append(module_doc)
        md.append("\n\n")

    # Each type
    for item in items:
        md.append(f"\\newpage\n\n")
        md.append(f"## {item['name']}\n\n")
        md.append(f"*{item['kind'].capitalize()}*\n\n")

        # Documentation
        if item['doc']:
            md.append(item['doc'])
            md.append("\n\n")

        # Code definition
        md.append("### Definition\n\n")
        md.append("```rust\n")
        md.append(item['code'])
        md.append("\n```\n\n")

        # Methods if available
        if item['name'] in impls:
            md.append("### Methods\n\n")
            for method in impls[item['name']]:
                md.append(f"#### `{method['name']}`\n\n")
                if method['doc']:
                    md.append(method['doc'])
                    md.append("\n\n")
                md.append("```rust\n")
                md.append(method['signature'])
                md.append("\n```\n\n")

    return ''.join(md)


def main():
    script_dir = Path(__file__).parent
    rust_file = script_dir / 'src' / 'lib.rs'
    output_md = script_dir / 'api-spec.md'
    output_pdf = script_dir / 'api-spec.pdf'

    print(f"Extracting documentation from {rust_file}...")
    module_doc, items, impls = extract_docs_and_code(rust_file)

    print(f"Found {len(items)} documented types")

    print(f"Generating markdown to {output_md}...")
    markdown = generate_markdown(module_doc, items, impls)

    with open(output_md, 'w') as f:
        f.write(markdown)

    print(f"Converting to PDF with pandoc...")
    try:
        subprocess.run([
            'pandoc',
            str(output_md),
            '-o', str(output_pdf),
            '--pdf-engine=pdflatex',
            '-V', 'geometry:margin=1in,landscape',
            '-V', 'fontsize=11pt',
            '--highlight-style=tango'
        ], check=True)
        print(f"✓ PDF generated: {output_pdf}")
    except FileNotFoundError:
        print("Warning: pandoc not found. Install with: sudo apt install pandoc texlive-latex-base texlive-latex-extra")
        print(f"Markdown file available at: {output_md}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc: {e}")
        print(f"Markdown file available at: {output_md}")


if __name__ == '__main__':
    main()
