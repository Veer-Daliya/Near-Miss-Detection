# Documentation

This folder contains the research paper and presentation materials for the Near-Miss Detection project.

## Contents

```
docs/
├── paper/
│   └── near_miss_detection.tex    # Research paper (LaTeX)
├── presentation/
│   ├── near_miss_presentation.tex       # Presentation with speaker notes
│   ├── near_miss_presentation_slides.tex # Slides only (audience version)
│   └── SPEAKER_SCRIPT.md                # Detailed speaker script
├── Makefile                       # Build automation
└── README.md                      # This file
```

## Compiling the PDFs

### Option 1: Local Compilation (Recommended)

**Prerequisites:** Install a LaTeX distribution:

```bash
# macOS (using Homebrew)
brew install --cask mactex-no-gui

# Ubuntu/Debian
sudo apt-get install texlive-full

# Windows
# Download and install MiKTeX: https://miktex.org/
```

**Compile all documents:**

```bash
cd docs
make all
```

**Or compile individually:**

```bash
make paper         # Research paper only
make presentation  # Presentation with speaker notes
make slides        # Slides only (no notes)
```

**Clean auxiliary files:**

```bash
make clean
```

### Option 2: Overleaf (Online)

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project → "Upload Project"
3. Upload the contents of `paper/` or `presentation/` folder
4. Click "Recompile" to generate the PDF

### Option 3: Manual Compilation

```bash
# Paper
cd docs/paper
pdflatex near_miss_detection.tex
pdflatex near_miss_detection.tex  # Run twice for references

# Presentation with notes
cd docs/presentation
pdflatex near_miss_presentation.tex
pdflatex near_miss_presentation.tex

# Slides only
cd docs/presentation
pdflatex near_miss_presentation_slides.tex
pdflatex near_miss_presentation_slides.tex
```

## Output Files

After compilation:

| File | Description |
|------|-------------|
| `paper/near_miss_detection.pdf` | Research paper |
| `presentation/near_miss_presentation.pdf` | Presentation with speaker notes |
| `presentation/near_miss_presentation_slides.pdf` | Slides only (for audience) |

## Required LaTeX Packages

The documents use standard packages included in most LaTeX distributions:

**Paper:**
- `times`, `geometry`, `graphicx`, `amsmath`, `amssymb`
- `booktabs`, `hyperref`, `algorithm`, `algpseudocode`

**Presentation:**
- `beamer` with `metropolis` theme
- `tikz`, `pgfplots`, `fontawesome5`
- `booktabs`, `amsmath`

## Customization

### Changing Author Information

Edit the `\author{}` command in:
- `paper/near_miss_detection.tex` (line ~51)
- `presentation/near_miss_presentation.tex` (line ~44)
- `presentation/near_miss_presentation_slides.tex` (line ~40)

### Adjusting Presentation Notes

The presentation with notes shows speaker notes on a second screen. To change this behavior, modify line 36 in `near_miss_presentation.tex`:

```latex
% Show notes on second screen (default)
\setbeameroption{show notes on second screen=right}

% Or hide notes entirely
\setbeameroption{hide notes}

% Or show notes only (for practice)
\setbeameroption{show only notes}
```

## Speaker Script

For detailed presentation guidance, see `presentation/SPEAKER_SCRIPT.md`. This includes:

- Timing for each slide
- Key talking points
- Anticipated Q&A responses
- Presentation tips
