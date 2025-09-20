# Localization Notes and Known Limitations

## Overview

The GraphJudge Streamlit Pipeline supports multiple languages (English, Simplified Chinese, Traditional Chinese) through JSON-based localization files in the `locales/` directory.

## Supported Languages

- **English** (`en.json`) - Default language
- **Simplified Chinese** (`zh_CN.json`) - 简体中文
- **Traditional Chinese** (`zh_TW.json`) - 繁體中文

## Language Switching

Users can switch languages using the sidebar language selector: 🌐 Language / 語言

## Known Limitations

### 1. Streamlit Native Widgets

Some Streamlit native widgets contain hardcoded English text that cannot be localized:

#### File Upload Widget (`st.file_uploader()`)
- **Issue**: Contains hardcoded text like "Drag and drop file here", "Browse files", etc.
- **Impact**: These texts will always appear in English regardless of selected language
- **Workaround**: We provide localized labels and help text around the widget
- **Location**: `ui/components.py` - `display_input_section()` function

#### Other Native Elements
- Progress bars may show "%" symbol formatting
- Some error messages from Streamlit itself remain in English
- Default browser file dialog text

### 2. Dependencies and External Libraries

#### Plotly Visualizations
- Chart axis labels and tooltips use English by default
- Interactive graph legends may show English text
- **Mitigation**: We customize chart titles and labels where possible

#### Pyvis Network Graphs
- Network visualization controls remain in English
- Node interaction tooltips may show mixed languages

## Implementation Details

### File Structure
```
locales/
├── en.json      # English translations
├── zh_CN.json   # Simplified Chinese
└── zh_TW.json   # Traditional Chinese
```

### Key Sections
- `app` - Application header and metadata
- `sidebar` - Sidebar controls and status
- `input` - Text input and file upload
- `processing` - Pipeline progress messages
- `results` - Results display and analysis
- `errors` - Error messages and warnings
- `buttons` - Action buttons and controls

### Usage Pattern
```python
from utils.i18n import get_text

# Simple text
st.title(get_text('app.title'))

# Text with parameters
st.success(get_text('file_upload.success_message', encoding=encoding))
```

## Maintenance

### Adding New Languages
1. Create new JSON file in `locales/` directory
2. Copy structure from `en.json`
3. Translate all values (keep keys unchanged)
4. Update language selector in sidebar

### Adding New Text Keys
1. Add key to all language files
2. Use consistent key naming: `section.subsection.key`
3. Test with all languages enabled

### Quality Assurance
- Verify all keys exist in all language files
- Test language switching functionality
- Check text formatting with different character sets
- Ensure proper encoding (UTF-8) for all files

## Best Practices

1. **Keep keys descriptive**: Use clear, hierarchical key names
2. **Maintain consistency**: Ensure all language files have same structure
3. **Test thoroughly**: Verify text display in all supported languages
4. **Document limitations**: Note any untranslatable elements
5. **Use parameters**: Utilize `{variable}` syntax for dynamic content

## Future Improvements

### Potential Enhancements
- Custom CSS to override some hardcoded Streamlit text
- Alternative UI components for better localization
- Server-side language detection
- RTL language support

### Streamlit Framework Limitations
- Cannot modify native widget text without custom CSS/JS
- Limited control over browser-generated text
- Some error messages come from Streamlit core

## Contact

For localization issues or translation improvements, please refer to the main project documentation.