# Documentation Guide

This guide explains how to build and contribute to Auralis documentation.

## Setup

1. Install documentation dependencies:

```bash
pip install auralis[docs]
# or directly:
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
```

## Building Documentation

### Local Development

Run the documentation server locally:

```bash
mkdocs serve
```

This will start a server at `http://127.0.0.1:8000`. The documentation will automatically reload when you make changes.

### Building Static Site

Generate static documentation:

```bash
mkdocs build
```

The built documentation will be in the `site` directory.

## Documentation Structure

```
docs/
├── index.md              # Main landing page
├── getting-started/      # Getting started guides
├── user-guide/          # User documentation
├── api/                 # API reference
├── advanced/           # Advanced topics
└── contributing/       # Contributing guides
```

## Writing Documentation

### Adding New Pages

1. Create your markdown file in the appropriate directory
2. Add it to the navigation in `mkdocs.yml`:

```yaml
nav:
  - Your Section:
    - your-page.md
```

### MkDocs Features

#### Code Blocks

```python
from auralis import TTS
tts = TTS()
```

#### Admonitions

```markdown
!!! note "Title"
    Content

!!! warning "Important"
    Warning content
```

#### API Documentation

```markdown
::: auralis.your.module.Class
    options:
      show_root_heading: true
      show_source: true
```

### Python Docstrings

Use Google-style docstrings for automatic API documentation:

```python
def function(arg: str) -> int:
    """Short description.
    
    Longer description if needed.

    Args:
        arg (str): Argument description.

    Returns:
        int: Return value description.

    Example:
        ```python
        result = function("test")
        ```
    """
```

## Best Practices

!!! tip "Documentation Tips"
    - Keep examples simple and focused
    - Use consistent formatting
    - Include practical code examples
    - Add admonitions for important notes
    - Cross-reference related documentation
    - Test code examples

## Building for Production

1. Update version numbers if needed
2. Build documentation:

```bash
mkdocs build --clean
```

3. Check the built site in `site/` directory
4. Deploy (if using GitHub Pages):

```bash
mkdocs gh-deploy
```

## Common Issues

!!! warning "Troubleshooting"
    - **Missing pages**: Check `mkdocs.yml` navigation
    - **Broken links**: Use relative paths
    - **Failed builds**: Verify Python package is installed
    - **Style issues**: Check Material theme documentation 