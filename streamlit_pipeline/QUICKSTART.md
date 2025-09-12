# GraphJudge Streamlit Application - Quick Start Guide

## 🚀 Getting Started

The GraphJudge Streamlit application provides a user-friendly web interface for the three-stage knowledge graph construction pipeline:

1. **Entity Extraction** 🔍 - Extract entities from Chinese text using GPT-5-mini
2. **Triple Generation** 🔗 - Generate knowledge graph triples from entities  
3. **Graph Judgment** ⚖️ - Evaluate triple quality using Perplexity AI

## 📋 Prerequisites

- Python 3.8 or higher
- Valid API keys for OpenAI and Perplexity

## 🛠️ Installation

1. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Optional Visualization Dependencies** (recommended):
   ```bash
   # Option A: Use the helper script
   python install_optional.py
   
   # Option B: Install manually
   pip install plotly>=5.0.0
   ```
   
   **Note**: Without Plotly, the app will work but show text-based displays instead of interactive charts.

3. **Configure API Keys** (create `.env` file):
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key
   
   # Perplexity Configuration  
   PERPLEXITY_API_KEY=your_perplexity_api_key
   
   # Optional: Azure OpenAI (alternative to standard OpenAI)
   # AZURE_OPENAI_KEY=your_azure_key
   # AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   ```

## 🎯 Running the Application

### Option 1: Direct Streamlit Command
```bash
streamlit run app.py
```

### Option 2: Using the Startup Script
```bash
python run_app.py
```

The application will be available at `http://localhost:8501`

## 📖 Using the Application

1. **Input Text**: Paste your Chinese text in the input area
2. **Click "开始处理"**: Start the three-stage pipeline
3. **Monitor Progress**: Watch real-time progress indicators
4. **View Results**: Examine extracted entities, triples, and final judgments
5. **Export Results**: Download results as JSON or CSV

## 🎨 Features

- **📝 Chinese Text Input**: Optimized for classical Chinese literature
- **📊 Real-time Progress**: Visual progress indicators for each stage
- **🔍 Detailed Results**: Stage-by-stage result visualization
- **📈 Knowledge Graph Viz**: Interactive network visualization
- **💾 Export Options**: JSON and CSV export functionality
- **❌ Error Recovery**: User-friendly error messages and suggestions
- **📋 Session History**: Track multiple processing runs

## 🔧 Configuration

The application uses these default settings:
- **Entity Model**: GPT-5-mini
- **Triple Model**: GPT-5-mini  
- **Judgment Model**: Perplexity Sonar-reasoning
- **Temperature**: 0.0 (deterministic)
- **Timeout**: 60 seconds per API call
- **Max Retries**: 3 attempts

Adjust these in the sidebar configuration panel.

## 📚 Example Input

Try this sample Chinese text:
```
红楼梦是清代作家曹雪芹创作的章回体长篇小说。小说以贾宝玉、林黛玉、薛宝钗三人的爱情婚姻悲剧为核心，以贾、史、王、薛四大家族的兴衰史为轴线，浓缩了整个封建社会的时代内容。
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory
2. **API Errors**: Verify your API keys in the `.env` file
3. **Module Not Found**: Install missing dependencies with pip
4. **Encoding Issues**: Ensure your terminal supports UTF-8

### Getting Help

- Check the sidebar for API status indicators
- Use the debug mode in sidebar configuration
- Review error messages for specific recovery suggestions

## 🏗️ Architecture

The application follows this structure:
```
streamlit_pipeline/
├── app.py                 # Main Streamlit application
├── core/                  # Refactored pipeline modules
│   ├── pipeline.py        # Pipeline orchestrator
│   ├── entity_processor.py
│   ├── triple_generator.py
│   └── graph_judge.py
├── ui/                    # UI components
│   ├── components.py      # Reusable components
│   ├── display.py         # Result displays
│   └── error_display.py   # Error handling
└── utils/                 # Shared utilities
    ├── api_client.py      # API integration
    ├── error_handling.py  # Error management
    └── validation.py      # Input validation
```

## 🎉 Success!

If everything is working correctly, you should see:
- Clean Chinese interface with input area
- "开始处理" button becomes active when text is entered
- Progress indicators show during processing
- Results display with knowledge graph visualization
- Export options become available after completion

Happy knowledge graph building! 🧠📊